#include "mcpl.h"
#include <stdio.h>
#include <stdlib.h>     /* atof */

int main(int argc,char**argv) {

  if (argc!=4) {
    printf("Please supply cutoff energy, input and output filenames\n");
    return 1;
  }

  const char * infilename = argv[2];
  const char * outfilename = argv[3];
  const double cut = atof(argv[1]);
  printf("cut is %e MeV\n", cut);

  // Initialisation, open existing file and create output file handle. Transfer
  // all meta-data from existing file, and add an extra comment in the output
  // file to document the process:

  mcpl_file_t fi = mcpl_open_file(infilename);
  mcpl_outfile_t fo = mcpl_create_outfile(outfilename);
  mcpl_transfer_metadata(fi, fo);
  mcpl_hdr_add_comment(fo,"Applied custom filter to select neutrons with ekin<0.1MeV");

  //Loop over particles from input, only triggering mcpl_add_particle calls for
  //the chosen particles:

  const mcpl_particle_t* particle;
  while ( ( particle = mcpl_read(fi) ) ) {
    if ( particle->pdgcode == 2112 && particle->ekin < cut ) {
      mcpl_add_particle(fo,particle);
      //Note that a guaranteed non-lossy alternative to mcpl_add_particle(fo,particle)
      //would be mcpl_transfer_last_read_particle(fi,fo) which can work directly on
      //the serialised on-disk particle data.
    }

  }

  //Close up files:
  mcpl_close_outfile(fo);
  mcpl_close_file(fi);
}