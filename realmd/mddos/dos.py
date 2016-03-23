# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:46:44 2013

@author: xwangan
"""
import numpy as np
import csv
from numpy import exp, sqrt, pi
from realmd.information import error, warning
from realmd import _ac as ac


class DOS():
    def __init__(self,
                 md,
                 atom_types=None,
                 is_ai=False, # is average input
                 sample_l=500, # sample length for correlation
                 corr_l=400, # correlation length
                 smooth_tune=1.0,
                 time_step=1.0, # time step in md calculation(fs)
                 is_sum=True,# is average in the 3 dimensions
                 is_wac=False,
                 is_sm = False,
                 is_normalize=False,
                 out_file_name="dos.csv"):
        self.dos={}
        self._md = md
        self.time_step=(md.step_indices[1]-md.step_indices[0])*time_step
        self.is_wac=is_wac
        self.is_normalize=is_normalize
        self.is_ai=is_ai
        self.is_summation=is_sum
        self.num_dim = 3 if is_sum==False else 1
        self.is_sm = is_sm
        self.smooth_tune=smooth_tune
        self.dos_file_name=out_file_name
        if atom_types is None:
            self.atom_types=md.atom_types
        else:
            self.atom_types=atom_types
        self.species=np.unique(self.atom_types)
        self.num_species=len(np.unique(self.atom_types))
        self.num_atom=md.num_atom
        self.num_steps=md.num_steps
        self.sample_length=sample_l
        self.corr_length=corr_l
        self.check_settting()
        self.set_velocities()
        self.print_information()
        
    def print_information(self):
        print "#"*75
        print "## information for dos calculation process"
        if self.num_species is not None:
            print "number of species:%d" %self.num_species
        if self.num_atom is not None:
            print "total number of atoms: %d" %self.num_atom
        if self.num_corr is not None:
            print "number of segments divided for the whole series: %d" %self.num_corr
        if self.sample_length is not None:
            print "steps in each segments for auto correlation calculation: %d" %self.sample_length
        if self.corr_length is not None:
            print "correlation length(redundancy requirement for auto correlation):%d" %self.corr_length
        print 
        
    
    def check_settting(self):
        if self.num_steps<self.sample_length:
            error("number of total time steps %d is lower than sampling length %d!" 
                % (self.num_steps, self.sample_length))
        if self.corr_length >= self.sample_length:
            error("correlation length %d is larger than sampling length %d!"
                %(self.corr_length, self.sample_length))
        elif self.sample_length-self.corr_length<5:
            warning("correlation of the last few time steps may be incorrect!")
    
    def set_velocities(self):
        num_corr=self.num_steps//self.sample_length
        vs=self._md.atom_velocities[:num_corr*self.sample_length]
        self.num_corr = num_corr
        self.velocities = np.zeros((self.num_atom, num_corr, self.sample_length, 3), dtype="double")
        for i in range(self.num_atom):
            self.velocities[i] = vs[:,i].reshape((num_corr, self.sample_length, 3))
        # self.velocities = vs.reshape((num_corr, self.sample_length, self.num_atom,3))
        #velocities belonging to each set of correlation should be normalized to 0
        if self.is_ai:
            print "force each of the segments to have a mean velocity of 0"
            ave=np.average(self.velocities,axis=2)
            for v in np.arange(self.sample_length):
                self.velocities[:,:,v] -= ave
        
        
    def run_dos(self):
        correlation_atom=np.zeros((self.num_atom, self.corr_length, 3, 3),dtype="double")
        # correlation = np.zeros((self.num_atom, self.corr_length, 3),dtype="double")
        # reshape velocity to the shape of (num_corr, sample_len, num_aom, 3)
        try:
            import _at
            self.run_auto_correlation_c(correlation_atom)
        except ImportError:
            self.run_auto_correlation_py(correlation_atom)
        correlation = np.einsum("ijkk->ijk", correlation_atom)
        # for i in range(3):
        #     correlation[:,:,i] = correlation_atom[:,:,i,i]
        if self.is_summation:
            print "averaging over the three dimensions"
            self.correlation=np.zeros((self.num_species,self.corr_length), dtype="double")
            axis_for_average=(0,-1)
        else:
            print "output data for three individual dimenisons"
            self.correlation=np.zeros((self.num_species,self.corr_length,3), dtype="double")
            axis_for_average=0
        for t,typ in enumerate(np.unique(self.atom_types)):
            pos = np.where(self.atom_types==typ)
            # sum over atoms belonging to one specie
            self.correlation[t]=np.sum(correlation[pos],axis=axis_for_average)
        if self.is_sm:
            self.gaussion_smoothing()
        cor0=np.sum(correlation[:,0], axis=axis_for_average)
        for i in np.arange(self.corr_length):
            self.correlation[:,i] /= cor0 # normalization over <x(0)*x(0)>
        self.fourier_transform()
        if self.is_wac:
            self.write_correlation()

        self.write_dos()
        import matplotlib.pyplot as plt
        plt.plot(self.correlation[0])
        plt.figure()
        plt.plot(self.correlation[0].cumsum() * self.time_step)
        plt.show()
        
        
    def run_auto_correlation_py(self,correlation_atoms):
        print "running auto correlation in python language..."
        for a in range(self.num_atom):
            correlation=np.zeros((self.num_corr, self.corr_length, 3, 3),dtype="double")
            self.run_auto_correlation_for_atom(a,correlation)
            correlation_atoms[a]=np.average(correlation, axis=0)
            
    def run_auto_correlation_c(self,correlation_atoms):
        print "running auto correlation in C language..."
        ac.auto_correlation_all_atoms(correlation_atoms,self.velocities,3)
    
    def run_auto_correlation_for_atom(self,i,correlation):
        v=self.velocities[i]
        for j in np.arange(self.num_corr):
            for k in np.arange(self.corr_length):
                cor_time=self.sample_length-k
                for m in np.arange(cor_time):
                    correlation[j,k] += np.outer(v[j,m], v[j,m+k])
                correlation[j,k] /= cor_time

    def gaussion_smoothing(self,sigma=None):
        """Apply gaussian smotting, correlation[num_type][corr_length][3]"""
        print("## Gaussian smoothing...");
        corr_length = self.corr_length
        if sigma is None:
            corr = self.correlation
            oscillation=np.max(np.abs(corr[:,:-1]-corr[:,1:]),axis=1)
            peak=np.max(np.abs(corr),axis=1)
            oscillation /= peak
            sigma= corr_length/(5.0*oscillation*len(corr)*self.smooth_tune) # 15.0 has been tuned for many times 
            print "sigma:"
            print sigma
        for i in np.arange(corr_length):
            self.correlation[:,i] *= exp(-i*i/(2*sigma*sigma))/(sigma*sqrt(2*pi))
            

    
    def fourier_transform(self):
        print "#"*75
        print "fourier transform for dos calculation"
        N=self.corr_length
        delta_t=self.time_step
        delta_f=1/(delta_t*N) # delta frequency
        dos=np.fft.fft(self.correlation,axis=1)[:,:N/2]/(N/2) # over the first axis(correlation length)
        dos=np.abs(dos)
        frequency=np.arange(N/2)*delta_f*1e3 # 1e3: 10^15Hz to THz
        #normalize dos
        if self.is_normalize:
            print "##normalizing dos"
            minimum=np.min(dos,axis=1)
            for f in np.arange(len(frequency)):
                dos[:,f] -= minimum
            total_dos=np.trapz(dos, frequency,axis=1).sum(axis=0)
            for f in np.arange(len(frequency)):
                dos[:,f] /= total_dos
        self.dos.setdefault("frequency",frequency)
        self.dos.setdefault("dos",dos) #only the first half counts
        
        
    
    def write_correlation(self,filename="auto_correlation.csv"):
        print "#"*75
        print "writing auto correlation data to file %s" %filename
        o=file(filename,'wb')
        csvo=csv.writer(o)
        num_species=self.num_species
        species=map(str,np.unique(self.atom_types))
        if self.is_summation:
            csvo.writerow(["time step","auto correlation"])
            csvo.writerow(['species']+list(species))
        else:
            csvo.writerow(["time step"]+
                ["auto correlation(x)"] +[""]*(num_species-1)+
                ["auto correlation(y)"] +[""]*(num_species-1)+
                ["auto correlation(z)"] +[""]*(num_species-1))
            csvo.writerow(['species']+list(species) * 3)
        for i in np.arange(self.corr_length):
            flat=self.correlation[:,i].swapaxes(0,-1).flatten() # flattening
            csvo.writerow([i]+list(flat))
        del csvo
        o.close()
                
    def write_dos(self):
        print "#"*75
        print "writing dos data to file %s" %self.dos_file_name
        o=file(self.dos_file_name,'wb')
        csvo=csv.writer(o)
        num_species=self.num_species
        species=map(str,np.unique(self.atom_types))
        if self.is_summation:
            csvo.writerow(["frequency(THz)","dos"])
            csvo.writerow(['species']+list(species))
        else:
            csvo.writerow(["frequency(THz)"]+
                ["dos(x)"]+[""]*(num_species-1)+
                ["dos(y)"]+[""]*(num_species-1)+
                ["dos(z)"]+[""]*(num_species-1))
            csvo.writerow(['species']+list(species)*3)
        for i in np.arange(self.corr_length//2):
            flat=np.absolute(self.dos['dos'][:,i]).swapaxes(0,-1).flatten()
            csvo.writerow([self.dos["frequency"][i]]+list(flat))
        del csvo
        o.close()
    
    def plot(self,is_plot=False, is_save=False):
        try:
            import matplotlib.pyplot as plt
            freq=self.dos["frequency"]
            dos=np.swapaxes(np.atleast_3d(self.dos["dos"]),0,1)
            for d in np.arange(dos.shape[-1]): # dimension
                plt.figure(num=d, figsize=(16,12))
                for s in np.arange(self.num_species):
                    plt.plot(freq, dos[:,s,d],linewidth=3,label="specie %s"%str(self.species[s]))
                plt.xlabel("Frequency(THz)",fontsize=18)
                plt.ylabel("DOS(Arbitary unit)",fontsize=18)
                plt.title("phonon density of states", fontsize=24)
                plt.legend(fontsize=14)
                if is_save:
                    plt.savefig("dos_%d.pdf"%d)
                if is_plot:
                    plt.show()    
        except ImportError:
            warning("lib matplotlib is not implemented, continuing anyway...")