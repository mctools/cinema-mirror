project(prompt_bin)


file(GLOB BIN "${PROJECT_SOURCE_DIR}/*.cc" "${PROJECT_SOURCE_DIR}/*.c")
foreach(TS ${BIN})
  get_filename_component(EXCNAME  ${TS}  NAME_WE)
  add_executable( ${EXCNAME} ${TS})
  target_link_libraries( ${EXCNAME} pthread prompt_core)
  # add_test(NAME ${EXCNAME} COMMAND ${EXCNAME} )
endforeach(TS ${BIN})
