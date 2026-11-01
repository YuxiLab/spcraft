import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { createWriteStream, existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const variantIndex = process.argv.indexOf('--variant');
const variant = variantIndex < 0 ? 'right-looking' : process.argv[variantIndex + 1];
if (!['right-looking', 'left-looking', 'right-looking-8x8', 'left-looking-8x8', 'up-looking-8x8'].includes(variant)) throw new Error('--variant must be right-looking, left-looking, right-looking-8x8, left-looking-8x8, or up-looking-8x8.');
const symbolic = variant.endsWith('-8x8');
const basename = symbolic ? variant.replace('-8x8','-lu-8x8') : `${variant}-lu`;
const html = join(root, `docs/solver/visuals/${basename}.html`);
const scratch = join(root, 'docs/obj', variant === 'right-looking' ? 'lu-video' : `${basename}-video`);
const output = join(root, `docs/video/${basename}.mp4`);
const frameDirectory = join(scratch, 'frames');
const fps = Number(process.env.LU_VIDEO_FPS || 24);
if (!Number.isFinite(fps) || fps < 1 || fps > 60) throw new Error('LU_VIDEO_FPS must be between 1 and 60.');

function executable(name) {
  const result = spawnSync('which', [name], { encoding: 'utf8' });
  return result.status === 0 ? result.stdout.trim() : null;
}
const chrome = process.env.CHROME_PATH || [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  executable('chromium'), executable('chromium-browser'), executable('google-chrome'),
].find(candidate => candidate && existsSync(candidate));
if (!chrome) throw new Error('Install Chrome/Chromium or set CHROME_PATH to its executable.');
const ffmpeg = executable('ffmpeg');
if (!ffmpeg) throw new Error('Install FFmpeg before rendering the video.');

await mkdir(scratch, { recursive: true });
await mkdir(dirname(output), { recursive: true });
const profile = await mkdtemp(join(scratch, 'chrome-profile-'));
const browserLog = createWriteStream(join(scratch, 'chrome.log'));
const browser = spawn(chrome, [
  '--headless=new', '--disable-gpu', '--disable-background-networking',
  '--disable-extensions', '--disable-sync', '--no-first-run', '--no-default-browser-check',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });
browser.stderr.pipe(browserLog);
let connection, encoder;
let browserExited = false;
browser.on('exit', () => { browserExited = true; });

try {
  let port;
  for (let attempt = 0; attempt < 100; ++attempt) {
    if (browserExited) throw new Error(`Chrome exited before its renderer became available; see ${scratch}/chrome.log.`);
    try {
      port = Number((await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]);
      break;
    } catch { await new Promise(resolve => setTimeout(resolve, 100)); }
  }
  if (!port) throw new Error('Timed out starting the isolated HTML renderer.');
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const page = targets.find(target => target.type === 'page');
  if (!page) throw new Error('Chrome did not create a rendering page.');
  connection = new WebSocket(page.webSocketDebuggerUrl);
  await once(connection, 'open');
  let nextId = 0;
  const pending = new Map();
  connection.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (!message.id) return;
    const callback = pending.get(message.id);
    if (!callback) return;
    pending.delete(message.id);
    if (message.error) callback.reject(new Error(message.error.message));
    else callback.resolve(message.result);
  });
  function call(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++nextId;
      pending.set(id, { resolve, reject });
      connection.send(JSON.stringify({ id, method, params }));
    });
  }
  async function evaluate(expression) {
    const result = await call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  }
  await call('Page.enable');
  await call('Runtime.enable');
  await call('Page.navigate', { url: pathToFileURL(html).href });
  let ready = false;
  for (let attempt = 0; attempt < 100; ++attempt) {
    ready = await evaluate('Boolean(window.luAnimation)');
    if (ready) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  if (!ready) throw new Error('The HTML animation failed to initialize.');
  // Screenshot the HTML canvas at its native 1080p size, without page controls.
  await call('Emulation.setDeviceMetricsOverride', { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
  await evaluate(`document.head.insertAdjacentHTML('beforeend', '<style>main{max-width:none;margin:0;padding:0}canvas{width:1920px;height:1080px;box-shadow:none}.controls,main>p{display:none}</style>'); document.fonts.ready`);
  async function capture(time) {
    await evaluate(`window.luAnimation.render(${time})`);
    const screenshot = await call('Page.captureScreenshot', {
      format: 'png', clip: { x: 0, y: 0, width: 1920, height: 1080, scale: 1 }, captureBeyondViewport: false,
    });
    return Buffer.from(screenshot.data, 'base64');
  }
  const checked = await evaluate('window.luAnimation.selfCheck()');
  const scenes = await evaluate('window.luAnimation.scenes');
  await writeFile(join(scratch, 'arithmetic-check.json'), JSON.stringify(checked, null, 2) + '\n');
  await writeFile(join(scratch, 'scenes.json'), JSON.stringify(scenes, null, 2) + '\n');
  console.log(`Verified ${checked.scalarUpdates} scalar updates, ${checked.multipliers} multipliers, and L*U=A.`);

  const firstUpdate=scenes.find(scene=>scene.kind==='update');
  const focus=scenes.find(scene=>scene.kind==='update'&&scene.i===5&&scene.j===6&&(['left-looking-8x8','up-looking-8x8'].includes(variant)?scene.p===2:scene.k===2))||firstUpdate;
  const previews = symbolic ? [
    ['intro', 1],
    ...['pivot', 'multiplier', 'update'].map(kind => {
      const scene=scenes.find(scene=>scene.kind===kind);
      return [kind,scene.start+scene.seconds/2];
    }),
    ...['before','after'].flatMap((phase,index)=>{
      const fraction=index===0?0.25:0.8;
      return [
        [`update-${phase}`,firstUpdate.start+firstUpdate.seconds*fraction],
        [`approved-${phase}`,focus.start+focus.seconds*fraction],
      ];
    }),
    ['complete', checked.duration-2],
  ] : [
    ['intro', 2],
    ['multiplier', scenes.find(scene => scene.kind === 'multiplier').start + 1.8],
    ['update', scenes.find(scene => scene.kind === 'update').start + 1.6],
    ['packed', scenes.find(scene => scene.kind === 'packed').start + 3],
    ['verify', scenes.find(scene => scene.kind === 'verify').start + 3],
    ['recap', scenes.find(scene => scene.kind === 'recap').start + 3],
  ];
  if (variant === 'left-looking') {
    previews.push(['source', scenes.find(scene => scene.kind === 'source' && scene.k === 1 && scene.j === 2).start + 1]);
  }
  if (['left-looking-8x8','up-looking-8x8'].includes(variant)) {
    for (const kind of ['load','source','commit']) {
      const scene=scenes.find(scene=>scene.kind===kind&&scene.k===focus.k&&(kind!=='source'||scene.p===2));
      previews.push([kind,scene.start+scene.seconds/2]);
    }
  }
  for (const [name, time] of previews) {
    await writeFile(join(scratch, `${name}.png`), await capture(time));
  }
  if (process.argv.includes('--preview')) {
    console.log(`Preview frames saved in ${scratch}`);
  } else {
    const encodeLog = createWriteStream(join(scratch, 'ffmpeg.log'));
    encoder = spawn(ffmpeg, [
      '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'image2pipe', '-framerate', String(fps),
      '-vcodec', 'png', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
      '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', output,
    ], { stdio: ['pipe', 'ignore', 'pipe'] });
    encoder.stderr.pipe(encodeLog);
    const encoding = once(encoder, 'exit');
    const frames = Math.ceil(checked.duration * fps);
    await mkdir(frameDirectory, { recursive: true });
    for (let frame = 0; frame < frames; ++frame) {
      const time = Math.min(checked.duration, frame/fps);
      const png = await capture(time);
      await writeFile(join(frameDirectory, `frame-${String(frame).padStart(6, '0')}.png`), png);
      if (!encoder.stdin.write(png)) await once(encoder.stdin, 'drain');
      if (frame % (fps * 10) === 0) console.log(`Rendered ${frame}/${frames} frames (${Math.floor(time)}s/${Math.ceil(checked.duration)}s).`);
    }
    encoder.stdin.end();
    const [code] = await encoding;
    if (code !== 0) throw new Error(`FFmpeg failed; see ${scratch}/ffmpeg.log.`);
    await writeFile(join(scratch, 'capture-manifest.json'), JSON.stringify({ html, output, frameDirectory, frames, fps,
      duration: frames / fps, width: 1920, height: 1080, captureMethod: 'Chrome Page.captureScreenshot' }, null, 2) + '\n');
    console.log(`Saved ${output} (${checked.duration.toFixed(1)} seconds, ${fps} fps, 1920x1080).`);
  }
} finally {
  connection?.close();
  if (encoder && encoder.exitCode === null) encoder.kill('SIGTERM');
  if (!browserExited) {
    const exited = once(browser, 'exit');
    browser.kill('SIGTERM');
    await exited;
  }
  browserLog.end();
  await rm(profile, { recursive: true, force: true });
}
