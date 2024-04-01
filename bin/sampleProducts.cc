/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

static char const *description = "Loops over each reaction does 1 product sampling at various projectile energies\nstarting at the reaction's threshold energy.";

#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <set>
#include <stdarg.h>

#include "MCGIDI.hpp"
#include "PTCentralData.hh"

namespace pt = Prompt;


/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/


#include <stdarg.h>

#include <string>
#include <vector>
#include <stdexcept>

class argvOption2 {

    public:
        std::string m_name;     // Must include all leading '-'s (E.g., "-v", "--target").
        int m_counter;
        bool m_needsValue;
        std::vector<int> m_indices;
        std::string m_descriptor;

        argvOption2( std::string const &a_name, bool a_needsValue, std::string const &a_descriptor = "" );

        bool present( ) const { return( m_counter > 0 ); }
        std::string zeroOrOneOption( char **argv, std::string const &a_default = "" );
        long asLong( char **a_argv, long a_default = 0 );
        double asDouble( char **a_argv, double a_default = 0.0 );
        void help( );
        void print( );
};

class argvOptions2 {

    public:
        std::string m_codeName;
        std::string m_descriptor;
        std::vector<argvOption2> m_options;
        std::vector<int> m_arguments;

        argvOptions2( std::string const &a_codeName, std::string const &a_descriptor = "" );

        int size( ) { return( static_cast<int>( m_options.size( ) ) ); }
        void add( argvOption2 const &a_option ) { m_options.push_back( a_option ); }
        void parseArgv( int argc, char **argv );
        argvOption2 *find( std::string const &a_name );
        long asLong( char **argv, int argumentIndex );
        double asDouble( char **argv, int argumentIndex );
        void help( );
        void print( );
};

long asLong2( char const *a_chars );
double asDouble2( char const *a_chars );
std::string doubleToString2( char const *format, double value );
std::string longToString2( char const *format, long value );
void MCGIDI_test_rngSetup( unsigned long long a_seed );
double float64RNG64( void *a_dummy );

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <iomanip>


#define PRINT_NAME_WIDTH 20

static unsigned long long state;
static unsigned long long a_factor = 0x27bb2ee687b0b0fd;
static unsigned long long b_addend = 0xb504f32d;
static double stateToDoubleFactor;

/*
=========================================================
*/
long asLong2( char const *a_chars ) {

    char *end_ptr;
    long value = strtol( a_chars, &end_ptr, 10 );

    while( isspace( *end_ptr  ) ) ++end_ptr;
    std::string msg( "ERROR: " );
    if( *end_ptr != 0 ) throw std::runtime_error( msg + a_chars + " is not a valid integer." );

    return( value );
}
/*
=========================================================
*/
double asDouble2( char const *a_chars ) {

    char *end_ptr;
    double value = strtod( a_chars, &end_ptr );

    while( isspace( *end_ptr  ) ) ++end_ptr;
    std::string msg( "ERROR: " );
    if( *end_ptr != 0 ) throw std::runtime_error( msg + a_chars + " is not a valid integer." );

    return( value );
}
/*
=========================================================
*/
std::string longToString2( char const *format, long value ) {

    char Str[256];

    sprintf( Str, format, value );
    std::string valueAsString( Str );

    return( valueAsString );
}
/*
=========================================================
*/
std::string doubleToString2( char const *format, double value ) {

    char Str[256];

    sprintf( Str, format, value );
    std::string valueAsString( Str );

    return( valueAsString );
}
/*
=========================================================
*/
void MCGIDI_test_rngSetup( unsigned long long a_seed ) {

    state = 0;
    --state;
    stateToDoubleFactor = 1.0 / state;

    state = a_seed;
}
/*
=========================================================
*/
double float64RNG64( void *a_dummy ) {

    state = a_factor * state + b_addend;
    return( stateToDoubleFactor * state );
}

/*
=========================================================
*/
argvOption2::argvOption2( std::string const &a_name, bool a_needsValue, std::string const &a_descriptor ) :
        m_name( a_name ),
        m_counter( 0 ),
        m_needsValue( a_needsValue ),
        m_descriptor( a_descriptor ) {

}
/*
=========================================================
*/
std::string argvOption2::zeroOrOneOption( char **a_argv, std::string const &a_default ) {

    std::string msg( "ERROR: " );

    if( !m_needsValue ) throw std::runtime_error( msg + m_name + " does not have a value." );
    if( m_counter > 1 ) throw std::runtime_error( msg + m_name + " does not allow multiple arguments." );
    if( m_counter == 0 ) return( a_default );
    return( a_argv[m_indices[0]] );
}
/*
=========================================================
*/
long argvOption2::asLong( char **a_argv, long a_default ) {

    if( present( ) ) {
        std::string msg( "ERROR: " );
        char *end_ptr;
        std::string value_string = zeroOrOneOption( a_argv, "" );

        a_default = strtol( a_argv[m_indices[0]], &end_ptr, 10 );

        while( isspace( *end_ptr  ) ) ++end_ptr;
        if( *end_ptr != 0 ) throw std::runtime_error( msg + value_string + " is not a valid integer." );
    }

    return( a_default );
}
/*
=========================================================
*/
double argvOption2::asDouble( char **a_argv, double a_default ) {

    if( present( ) ) {
        std::string msg( "ERROR: " );
        char *end_ptr;
        std::string value_string = zeroOrOneOption( a_argv, "" );

        a_default = strtod( a_argv[m_indices[0]], &end_ptr );

        while( isspace( *end_ptr  ) ) ++end_ptr;
        if( *end_ptr != 0 ) throw std::runtime_error( msg + value_string + " is not a valid double." );
    }   
    
    return( a_default );
}
/*
=========================================================
*/
void argvOption2::help( ) {

    std::cout << "    " << std::left << std::setw( PRINT_NAME_WIDTH ) << m_name;
    if( m_needsValue ) {
        std::cout << " VALUE  "; }
    else {
        std::cout << "        ";
    }
    std::cout << m_descriptor << std::endl;
}
/*
=========================================================
*/
void argvOption2::print( ) {

    std::cout << std::setw( PRINT_NAME_WIDTH ) << m_name;
    for( std::vector<int>::iterator iter = m_indices.begin( ); iter != m_indices.end( ); ++iter ) std::cout << " " << *iter;
    std::cout << std::endl;
}

/*
=========================================================
*/
argvOptions2::argvOptions2( std::string const &a_codeName, std::string const &a_descriptor ) :
        m_codeName( a_codeName ),
        m_descriptor( a_descriptor ) {

    add( argvOption2( "-h", false, "Show this help message and exit." ) );
}
/*
=========================================================
*/
void argvOptions2::parseArgv( int argc, char **argv ) {

    for( int iargc = 1; iargc < argc; ++iargc ) {
        std::string arg( argv[iargc] );

        if( arg == "-h" ) help( );
        if( arg[0] == '-' ) {
            int index = 0;

            for( ; index < size( ); ++index ) {
                argvOption2 &option = m_options[index];

                if( option.m_name == arg ) {
                    ++option.m_counter;
                    if( option.m_needsValue ) {
                        ++iargc;
                        if( iargc == argc ) {
                            std::string msg( "ERROR: option '" );

                            throw std::runtime_error( msg + arg + "' has no value." );
                        }
                        option.m_indices.push_back( iargc );
                    }
                    break;
                }
            }

            if( index == size( ) ) {
                std::string msg( "ERROR: invalid option '" );
                throw std::runtime_error( msg + arg + "'." );
            } }
        else {
            m_arguments.push_back( iargc );
        }
    }
}
/*
=========================================================
*/
argvOption2 *argvOptions2::find( std::string const &a_name ) {

    for( std::vector<argvOption2>::iterator iter = m_options.begin( ); iter != m_options.end( ); ++iter ) {
        if( iter->m_name == a_name ) return( &(*iter) );
    }
    return( nullptr );
}
/*
=========================================================
*/
long argvOptions2::asLong( char **argv, int argumentIndex ) {

    return( ::asLong2( argv[m_arguments[argumentIndex]] ) );
}
/*
=========================================================
*/
double argvOptions2::asDouble( char **argv, int argumentIndex ) {

    return( ::asDouble2( argv[m_arguments[argumentIndex]] ) );
}
/*
=========================================================
*/
void argvOptions2::help( ) {

    std::cout << std::endl << "Usage:" << std::endl;
    std::cout << "    " << m_codeName << std::endl;
    if( m_descriptor != "" ) {
        std::cout << std::endl << "Description:" << std::endl;
        std::cout << "    " << m_descriptor << std::endl;
    }

    if( m_options.size( ) > 0 ) {
        std::cout << std::endl << "Options:" << std::endl;
        for( std::vector<argvOption2>::iterator iter = m_options.begin( ); iter != m_options.end( ); ++iter ) iter->help( );
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/
void argvOptions2::print( ) {

    std::cout << "Arugment indices:";
    for( std::vector<int>::iterator iter = m_arguments.begin( ); iter != m_arguments.end( ); ++iter ) std::cout << " " << *iter;
    std::cout << std::endl;
    for( std::vector<argvOption2>::iterator iter = m_options.begin( ); iter != m_options.end( ); ++iter ) iter->print( );
}

void main2( int argc, char **argv );
/*
=========================================================
*/
int main( int argc, char **argv ) {

    try {
        main2( argc, argv ); }
    catch (std::exception &exception) {
        std::cerr << exception.what( ) << std::endl;
        exit( EXIT_FAILURE ); }
    catch (char const *str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE ); }
    catch (std::string &str) {
        std::cout << str << std::endl;
        exit( EXIT_FAILURE );
    }

    exit( EXIT_SUCCESS );
}
/*
=========================================================
*/

#include "PromptCore.hh"

#include "PTSingleton.hh"
#include "PTRandCanonical.hh"

inline double getRandNumber(void *obj) 
{
  return Prompt::Singleton<Prompt::SingletonPTRand>::getInstance().generate();
}



void main2( int argc, char **argv ) {

    PoPI::Database m_pops(pt::Singleton<pt::CentralData>::getInstance().getGidiPops());
    GIDI::Protare *gidiprotare;
    GIDI::Transporting::Particles particles;


    double energyDomainMax = 20.0;

    std::set<int> reactionsToExclude;
    LUPI::StatusMessageReporting smr1;
    GIDI::Construction::PhotoMode photo_mode = GIDI::Construction::PhotoMode::nuclearAndAtomic;

    std::cerr << "    " << __FILE__;
    for( int i1 = 1; i1 < argc; i1++ ) std::cerr << " " << argv[i1];
    std::cerr << std::endl;

    argvOptions2 argv_options( "sampleProducts", description );

    argv_options.add( argvOption2( "--pid", true, "The PoPs id of the projectile." ) );
    argv_options.add( argvOption2( "--tid", true, "The PoPs id of the target." ) );

    argv_options.add( argvOption2( "--electron", false, "If present and the protare is photo-atomic, electrons are transported." ) );

    argv_options.parseArgv( argc, argv );
    std::string mapFilename =  pt::Singleton<pt::CentralData>::getInstance().getGidiMap() ;

    std::string targetID = argv_options.find( "--tid" )->zeroOrOneOption( argv, "O16" );
    std::string ProjectileID = argv_options.find( "--pid" )->zeroOrOneOption( argv, "n" );

    GIDI::Transporting::DelayedNeutrons delayedNeutrons = GIDI::Transporting::DelayedNeutrons::on;


    // bool transportElectrons = argv_options.find( "--electron" )->present( );

    GIDI::Map::Map map( mapFilename, m_pops );


    GIDI::Construction::Settings construction( GIDI::Construction::ParseMode::all, photo_mode );
    gidiprotare = (GIDI::Protare *) map.protare( construction, m_pops, ProjectileID, targetID , "", "", false, false);

    GIDI::Styles::TemperatureInfos temperatures = gidiprotare->temperatures( );
    for( GIDI::Styles::TemperatureInfos::const_iterator iter = temperatures.begin( ); iter != temperatures.end( ); ++iter ) {
        std::cout << "label = " << iter->heatedCrossSection( ) << "  temperature = " << iter->temperature( ).value( ) << std::endl;
    }

    std::string label( temperatures[0].griddedCrossSection( ) );
    MCGIDI::Transporting::MC *MC = new MCGIDI::Transporting::MC( m_pops, gidiprotare->projectile( ).ID( ), &gidiprotare->styles( ), label, delayedNeutrons, energyDomainMax );
    MC->sampleNonTransportingParticles( false );
    MC->set_wantRawTNSL_distributionSampling( true );

    // GIDI::Transporting::Groups_from_bdfls groups_from_bdfls( "../../../GIDI/Test/bdfls" );
    // GIDI::Transporting::Fluxes_from_bdfls fluxes_from_bdfls( "../../../GIDI/Test/bdfls", 0 );

    GIDI::Transporting::Particle neutron( PoPI::IDs::neutron , GIDI::Transporting::Mode::MonteCarloContinuousEnergy);
    particles.add( neutron );

    GIDI::Transporting::Particle photon( PoPI::IDs::photon, GIDI::Transporting::Mode::MonteCarloContinuousEnergy );
    particles.add( photon );
    

    // if( transportElectrons && !particles.hasParticle( PoPI::IDs::electron ) ) {
    //     for( std::size_t protareIndex = 0; protareIndex < protare->numberOfProtares( ); ++protareIndex ) {
    //         GIDI::ProtareSingle *protareSingle = protare->protare( protareIndex );

    //         if( protareSingle->isPhotoAtomic( ) ) {
    //             GIDI::Transporting::Particle electron( PoPI::IDs::electron );
    //             particles.add( electron );
    //             break;
    //         }
    //     }
    // }

    MCGIDI::DomainHash domainHash( 4000, 1e-8, 20 );
    MCGIDI::Protare *MCProtare = MCGIDI::protareFromGIDIProtare( smr1, *gidiprotare, m_pops, *MC, particles, domainHash, temperatures, reactionsToExclude );
    delete gidiprotare;
    delete MC;

    MCGIDI::Vector<MCGIDI::Protare *> protares(1);
    protares[0]= MCProtare;
    MCGIDI::URR_protareInfos URR_protareInfos(protares);


    MCProtare->setUserParticleIndex( m_pops[PoPI::IDs::neutron], 1011 );
    MCProtare->setUserParticleIndex( m_pops["H2"], 10 );
    MCProtare->setUserParticleIndex( m_pops["B10"], 810 );
    MCProtare->setUserParticleIndex( m_pops[PoPI::IDs::photon], 11 );
    MCProtare->setUserParticleIndex( m_pops[PoPI::IDs::electron], 12 );

    MCGIDI::Sampling::Input input( false, MCGIDI::Sampling::Upscatter::Model::B );
    input.m_temperature = 2.58e-5;  // In keV/k;
    

    double ekin_MeV = 10; 
    int hashIndex = domainHash.index(ekin_MeV); 
    auto *m_products = new MCGIDI::Sampling::StdVectorProductHandler();


    for(unsigned i=0;i<100;i++)
    {
        double m_cacheGidiXS = MCProtare->crossSection( URR_protareInfos, hashIndex, input.m_temperature, ekin_MeV ); 

        int reactionIndex = MCProtare->sampleReaction( URR_protareInfos, hashIndex, input.m_temperature, ekin_MeV, m_cacheGidiXS, getRandNumber, nullptr );
        MCGIDI::Reaction const *reaction = MCProtare->reaction( reactionIndex );

        m_products->clear();
        reaction->sampleProducts( MCProtare, ekin_MeV, input, getRandNumber, nullptr, *m_products );
        std::cout << "ENDF MT" << reaction->ENDF_MT() <<  ", m_products->size() " <<  m_products->size() << std::endl;
        
        std::vector<MCGIDI::Sampling::Product> prod_n;
        double totalekin = 0;
        for( std::size_t i = 0; i < m_products->size( ); ++i ) 
        {
            if ((*m_products)[i].m_productIndex==11)
            prod_n.push_back((*m_products)[i]);
            
            totalekin += (*m_products)[i].m_kineticEnergy;

            // if(reaction->finalQ(ekin_MeV))
            std::cout <<"m_projectileMass " << input.m_projectileMass << ", "
            << (*m_products)[i].m_productIndex << " " 
            << reaction->finalQ(ekin_MeV) << " " 
            << (*m_products)[i].m_kineticEnergy << "\n";
        }

        printf("deposition %f\n\n", ekin_MeV+reaction->finalQ(ekin_MeV)-totalekin);

        // std::cout << " total neutrons " << prod_n.size() << "\n";

        // if MC.sampleNonTransportingParticles(true), many of the events are sampled in the centerOfMass
        // if(input.m_frame == GIDI::Frame::centerOfMass)
        //     PROMPT_THROW(NotImplemented, "GIDI::Frame::centerOfMass product is not yet implemented");
        
    }


    delete m_products;
    delete MCProtare;
}
