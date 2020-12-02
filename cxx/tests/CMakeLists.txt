project(testcxx)


file(GLOB TESTFILES "${PROJECT_SOURCE_DIR}/*.cc" "${PROJECT_SOURCE_DIR}/*.c")
foreach(TS ${TESTFILES})
  get_filename_component(EXCNAME  ${TS}  NAME_WE)
  add_executable( ${EXCNAME} ${TS} )
  target_link_libraries( ${EXCNAME} pthread PiXiu)
endforeach(TS ${TESTFILES})
