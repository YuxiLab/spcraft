# CMakeLists.txt for SPCraft Metaprogramming Examples

set(METAPROGRAMMING_EXAMPLES
    01_type_parameterization
    02_ownership_and_move_semantics
    03_concepts_and_constraints
    04_constexpr_dispatch
    05_expression_templates
    06_policy_based_design
)

foreach(example ${METAPROGRAMMING_EXAMPLES})
    add_executable(${example} ${example}.cpp)
    target_link_libraries(${example}
        PRIVATE
            SPCraft
            fmt::fmt
    )
    target_compile_features(${example} PRIVATE cxx_std_20)
endforeach()
