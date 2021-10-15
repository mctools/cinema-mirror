project(prompt_test)

add_library(test_main OBJECT test_main.cpp)

file(GLOB TESTFILES "${PROJECT_SOURCE_DIR}/unittest/*.cc" "${PROJECT_SOURCE_DIR}/unittest/*.c")
foreach(TS ${TESTFILES})
  get_filename_component(EXCNAME  ${TS}  NAME_WE)
  add_executable( ${EXCNAME} ${TS} $<TARGET_OBJECTS:test_main>)
  target_link_libraries( ${EXCNAME} pthread prompt_core)
  add_test(NAME ${EXCNAME} COMMAND ${EXCNAME} )
endforeach(TS ${TESTFILES})
