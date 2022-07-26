import os, json
from  spglib import get_spacegroup,standardize_cell
from phonopy.interface.vasp import read_vasp
from phonopy.structure.atoms import atom_data
import glob
from .AtomInfo import getAtomMassBC
from enum import Enum, unique

@unique
class QEType(Enum):
    Relax = 0
    Scf = 1

class Pseudo():
    def __init__(self):
        self.libpath=os.environ['PIXIUSSSP']
        if self.libpath is None:
            raise IOError('PIXIUSSSP enviroment is not set')
        self.pseudoinfo = json.load( open(self.libpath+'/SSSP_1.2.0_PBE_precision.json'))
        self.libpath += '/SSSP_precision_pseudos'


    def getPseudo(self, elementName, copyTo = None):
        ecutwfc=self.pseudoinfo[elementName]['cutoff_wfc']
        ecutrho=self.pseudoinfo[elementName]['cutoff_rho']
        pseudoName = self.pseudoinfo[elementName]['filename']
        return ecutwfc, ecutrho, pseudoName

    def linkvdWTable(self):
        os.link(self.libpath+'/vdW_kernel_table', './vdW_kernel_table')

    def qe_input(self, qeType):
        if qeType==QEType.Relax:
            qe_control= """ &control
                calculation = 'vc-relax'
                restart_mode='from_scratch'
                tprnfor = .true.
                max_seconds = 36000.0
                pseudo_dir = {ppath}
                disk_io = 'nowf'
                prefix='out'
             /
             &system
                ibrav = 0
                nat = {nat}
                ntyp = {ntyp}
                occupations='smearing'
                smearing='gauss'
                degauss=0.02
                ecutwfc = {ecutwfc}, ecutrho={ecutrho}
                {vdwOrmag}
             /
             &electrons
                conv_thr = 1.0d-12
                mixing_beta = 0.7
                mixing_mode = 'plain'
                diagonalization= 'david'
            /
            &IONS
            ion_dynamics= 'bfgs'
            /
            &CELL
           cell_dynamics = 'bfgs'
             /
            K_POINTS automatic
            {kp0} {kp1} {kp2} 0 0 0\n"""
            return qe_control

        elif qeType==QEType.Scf:
            qe_control= """ &control
                calculation = 'scf'
                restart_mode='from_scratch'
                tprnfor = .true.
                max_seconds = 36000.0
                pseudo_dir = {ppath}
                disk_io = 'nowf'
                prefix='out'
             /
             &system
                ibrav = 0
                nat = {nat}
                ntyp = {ntyp}
                occupations='smearing'
                smearing='gauss'
                degauss=0.02
                ecutwfc = {ecutwfc}, ecutrho={ecutrho}
                !input_dft  = 'vdw-df2'
                {vdwOrmag}
             /
             &electrons
                conv_thr = 1.0d-12
                mixing_beta = 0.7
                mixing_mode = 'plain'
                diagonalization= 'david'
             /
            K_POINTS automatic
            {kp0} {kp1} {kp2} 0 0 0\n"""
            return qe_control


    #quantum expresso make simple
    def qems(self, cell, simname, dim, kpt, qeType=QEType.Scf, usePrimitiveCell=True, vdW=False):
        metal_str="occupations='smearing'"

        qe_control=self.qe_input(qeType)
        atom_spec ="ATOMIC_SPECIES\n{}"
        vdwOrmag=None
        if vdW:
            vdwOrmag = """input_dft  = 'vdw-df2'"""
            self.linkvdWTable()
        else:
            vdwOrmag = """nspin = 2, starting_magnetization=1.0"""



        if qeType ==  QEType.Relax:
            lattice , positions, numbers_p = standardize_cell(cell, to_primitive=usePrimitiveCell, no_idealize=0, symprec=0.1)
        elif qeType == QEType.Scf:
            lattice , positions, numbers_p = cell

        spacegroup = get_spacegroup((lattice , positions, numbers_p), symprec=1e-5)

        elements=[]
        ele_num=[]
        tot_ele_num =0
        for num in numbers_p:
            tot_ele_num += 1
            syb = atom_data[num][1]
            if syb not in elements:
                elements.append(syb)
                ele_num.append(1)
            else :
                idx = elements.index(syb)
                ele_num[idx] += 1

        unit_vec = ' '.join(map(str, lattice[0])) + '\n'+ ' '.join(map(str, lattice[1])) + '\n'+' '.join(map(str, lattice[2])) + '\n'

        pos=''
        for i in range(tot_ele_num):
            pos+=atom_data[numbers_p[i]][1] + ' ' + ' '.join(map(str, positions[i])) + '\n'

        #build input
        element_mass = []
        pseudopotentials=[]
        max_ecutwfc=0.
        max_ecutrho=0

        for ele in elements:
            mass, _, _ = getAtomMassBC(ele)
            element_mass.append(mass)

            ecutwfc, ecutrho, pseudo = self.getPseudo(ele)
            pseudopotentials.append(pseudo)
            max_ecutwfc=max(max_ecutwfc,ecutwfc)
            max_ecutrho=max(max_ecutrho,ecutrho)


        defatom=''
        for i in range(len(elements)):
            defatom += elements[i]+' '+str(element_mass[i])+' '+pseudopotentials[i] + "\n"

        f=open(simname,'w')
        #relax
        f.write(qe_control.format(ppath="'"+self.libpath+"'", vdwOrmag=vdwOrmag, nat=tot_ele_num,ntyp=len(elements),kp0=kpt[0]*2,kp1=kpt[1]*2,kp2=kpt[2]*2,ecutwfc=max_ecutwfc, ecutrho=max_ecutrho))
        f.write(atom_spec.format(defatom))
        f.write('ATOMIC_POSITIONS crystal\n')
        f.writelines(pos)
        f.write('CELL_PARAMETERS angstrom\n')
        f.writelines(unit_vec)
        f.close()

        suclfiles=None

        if qeType==QEType.Scf:
            if os.system('phonopy --qe -d -v --dim="{dim1} {dim2} {dim3}" '.format(dim1=dim[0],dim2=dim[1],dim3=dim[2])+ " -c " + simname):
                raise IOError("can't create supercell")

            suclfiles= glob.glob1('.',"supercel*.in")
            if len(suclfiles)==0:
                raise IOError("phononpy didn't created any supercell")
            print ("generated", len(suclfiles)-1, "supercell files:")
            print (suclfiles)
            for sucl in suclfiles:
                f=open(sucl,'r+')
                content = f.read()
                f.seek(0,0)
                if usePrimitiveCell:
                    f.write('! primitive cell\n')
                else:
                    f.write('! conventional cell\n')
                f.write("! supercell {dim1} {dim2} {dim3}\n".format(dim1=dim[0],dim2=dim[1],dim3=dim[2]) )
                f.write(f'! kt {kpt[0]} {kpt[1]} {kpt[2]} \n')
                #scf calculation
                if vdW:
                    f.write(qe_control.format(ppath="'"+self.libpath+"'", vdwOrmag=vdwOrmag, nat=tot_ele_num*dim[0]*dim[1]*dim[2],ntyp=len(elements),kp0=kpt[0],kp1=kpt[1],kp2=kpt[2],atom=defatom
                        ,ecutwfc=max_ecutwfc, ecutrho=max_ecutrho) + content)
                else:
                    #magnetisation calculation is too slow, disabled for now
                    f.write(qe_control.format(ppath="'"+self.libpath+"'", vdwOrmag='' , nat=tot_ele_num*dim[0]*dim[1]*dim[2],ntyp=len(elements),kp0=kpt[0],kp1=kpt[1],kp2=kpt[2],atom=defatom
                        ,ecutwfc=max_ecutwfc, ecutrho=max_ecutrho) + content)
                f.close()

        return spacegroup, lattice , positions, elements, numbers_p
