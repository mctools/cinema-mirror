

#include "PTPython.hh"
#include "PTPhysicsFactory.hh"
#include "mcpl.h"

double pt_nccalNumDensity(const char *s)
{
    return Prompt::Singleton<Prompt::PhysicsFactory>::getInstance().nccalNumDensity(s);
}

void pt_merge_mcpl(const char* file_output, unsigned nfiles, const char ** files)
{
    mcpl_outfile_t fo = mcpl_merge_files( file_output, nfiles, files );
    mcpl_closeandgzip_outfile(fo);
    //fixme: consider mcpl_closeandgzip_outfile ??
}
