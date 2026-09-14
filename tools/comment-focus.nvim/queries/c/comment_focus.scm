; Match declaration names, not call sites or parameter names.
(function_declarator
  declarator: (identifier) @function)

(struct_specifier
  name: (type_identifier) @class)

(union_specifier
  name: (type_identifier) @class)

(enum_specifier
  name: (type_identifier) @class)
