#include "mcpl.h"
#include <stdio.h>
#include <stdlib.h>     /* atof */
#include <string>

int main(int argc,char**argv) {

  if (argc!=4) {
    printf("Please supply input, output filenames and cutoff energy in eV\n");
    return 1;
  }

  const char * infilename = argv[1];
  const char * outfilename = argv[2];
  const double cut = atof(argv[3])*1e-6;
  printf("cut is %e MeV\n", cut);

  // Initialisation, open existing file and create output file handle. Transfer
  // all meta-data from existing file, and add an extra comment in the output
  // file to document the process:

  mcpl_file_t fi = mcpl_open_file(infilename);
  mcpl_outfile_t fo = mcpl_create_outfile(outfilename);
  mcpl_transfer_metadata(fi, fo);
  mcpl_hdr_add_comment(fo, ("Applied custom filter to select neutrons with ekin" + std::string(argv[3]) + "eV").c_str() );

  //Loop over particles from input, only triggering mcpl_add_particle calls for
  //the chosen particles:
  u_int64_t n_par(0), n_par_removed(0);
  const mcpl_particle_t* particle;
  while ( ( particle = mcpl_read(fi) ) ) {
    n_par++;
    if ( particle->pdgcode == 2112 && particle->ekin < cut ) {
      mcpl_add_particle(fo,particle);
      //Note that a guaranteed non-lossy alternative to mcpl_add_particle(fo,particle)
      //would be mcpl_transfer_last_read_particle(fi,fo) which can work directly on
      //the serialised on-disk particle data.
      n_par_removed++;
    }

  }

  printf("particle %lu, after filtering %lu\n", n_par, n_par_removed);
  //Close up files:
  mcpl_close_outfile(fo);
  mcpl_close_file(fi);
}