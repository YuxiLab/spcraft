import { spawn, spawnSync } from 'node:child_process';
import { once } from 'node:events';
import { createWriteStream, existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const variantIndex = process.argv.indexOf('--variant');
const variant = variantIndex < 0 ? 'right-looking' : process.argv[variantIndex + 1];
if (!['right-looking', 'left-looking'].includes(variant)) throw new Error('--variant must be right-looking or left-looking.');
const html = join(root, `docs/solver/visuals/${variant}-lu.html`);
const scratch = join(root, 'docs/obj', variant === 'right-looking' ? 'lu-video' : `${variant}-lu-video`);
const output = join(root, `docs/video/${variant}-lu.mp4`);
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
  const checked = await evaluate('window.luAnimation.selfCheck()');
  const scenes = await evaluate('window.luAnimation.scenes');
  await writeFile(join(scratch, 'arithmetic-check.json'), JSON.stringify(checked, null, 2) + '\n');
  await writeFile(join(scratch, 'scenes.json'), JSON.stringify(scenes, null, 2) + '\n');
  console.log(`Verified ${checked.scalarUpdates} scalar updates, ${checked.multipliers} multipliers, and L*U=A.`);

  const previews = [
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
  for (const [name, time] of previews) {
    const png = await evaluate(`window.luAnimation.render(${time}); document.getElementById('lesson').toDataURL('image/png').split(',')[1]`);
    await writeFile(join(scratch, `${name}.png`), Buffer.from(png, 'base64'));
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
    for (let frame = 0; frame < frames; ++frame) {
      const time = Math.min(checked.duration, frame/fps);
      const png = await evaluate(`window.luAnimation.render(${time}); document.getElementById('lesson').toDataURL('image/png').split(',')[1]`);
      if (!encoder.stdin.write(Buffer.from(png, 'base64'))) await once(encoder.stdin, 'drain');
      if (frame % (fps * 10) === 0) console.log(`Rendered ${frame}/${frames} frames (${Math.floor(time)}s/${Math.ceil(checked.duration)}s).`);
    }
    encoder.stdin.end();
    const [code] = await encoding;
    if (code !== 0) throw new Error(`FFmpeg failed; see ${scratch}/ffmpeg.log.`);
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
