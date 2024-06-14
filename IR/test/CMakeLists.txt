configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(RVIR_TEST_DEPENDS
        FileCheck count not
        rv-opt
        )

add_lit_testsuite(check-all "Running the standalone regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${RVIR_TEST_DEPENDS}
        )
set_target_properties(check-all PROPERTIES FOLDER "Tests")
add_lit_testsuites(RVIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${RVIR_TEST_DEPENDS})