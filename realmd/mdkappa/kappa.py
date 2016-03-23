# -*- coding: utf-8 -*-
"""
Created on Sunday Sept 14 19:38:44 2013

@author: xwangan
"""
import numpy as np
import csv
from numpy import exp, sqrt, pi
from string import maketrans
from realmd.information import error, warning
from realmd.unit import fs2ps, kb, Angstrom, EV, ps, THz, bar, au_force, Bohr2Angstrom

# from outside import heat_flux
unit_conversion= 1 / Angstrom ** 3 / EV * ps * (Angstrom / ps * EV) ** 2 # conversion factor to W/m-K
class Kappa():
    def __init__(self,
                 md_cvfe, #c: coordinate, v: velocity, f: force and e: energy
                 temperature=300,
                 volume = None,
                 # atom_types=None,
                 is_ai=False, # is average input
                 sample_l=500, # sample length for correlation
                 corr_l=400, # correlation length
                 smooth_tune=1.0,
                 time_step=1.0, # time step in md calculation(fs)
                 is_diff = False,
                 is_sum=True,# is average in the 3 dimensions
                 is_wac=False,
                 is_sm = False,
                 is_normalize=False,
                 dirt=None, #direction tensor
                 out_file_name="dos.csv"):
        self.kappa_omega={}
        self._md_cvfe = md_cvfe
        self.temperature = temperature
        self.box_bounds = self._md_cvfe.lattice_bounds
        if self.box_bounds is not None:
            self.volume = np.prod(self.box_bounds[:,1] - self.box_bounds[:,0])
        elif volume is not None:
            self.volume = volume
        else:
            error("The volume is not set!")
        self.time_step=(md_cvfe.step_indices[1]-md_cvfe.step_indices[0]) * time_step * fs2ps # [ps]
        self.is_wac=is_wac
        self.is_diff = is_diff
        self.is_normalize=is_normalize
        self.is_ai=is_ai
        self.is_summation=is_sum
        self.num_dim = 3 if is_sum==False else 1
        self.is_sm = is_sm
        self.smooth_tune = smooth_tune
        self.dos_file_name = out_file_name
        self.dirt = dirt
        self.kappa=None
        self.num_atom = md_cvfe.num_atom
        self.num_steps = md_cvfe.num_steps
        self.sample_length = sample_l
        self.corr_length=corr_l
        self.conversion_factor = 1 / self.volume / kb / self.temperature ** 2 * unit_conversion
        self.check_settting()
        self.correlation_init()
        self.print_information()
        
    def print_information(self):
        print "#"*75
        print "## information for kappa calculation process"
        if self.num_atom is not None:
            print "total number of atoms: %d" %self.num_atom
        if self.num_corr is not None:
            print "number of segments divided for the whole series: %d" %self.num_corr
        if self.time_step is not None:
            print "time step: %.2f (fs)" %(self.time_step / fs2ps)
        if self.sample_length is not None:
            print "steps in each segments for auto correlation calculation: %d" %self.sample_length
        if self.corr_length is not None:
            print "correlation length(redundancy requirement for auto correlation):%d" %self.corr_length
        if self.dirt is not None:
            print "Thermal conductivity tensor: %s" % ("".join(map(str,self.dirt)).translate(maketrans("012", "xyz")))
        else:
            print "Thermal conductivity is averaged over 3 dimensions"
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
    
    def correlation_init(self):
        num_corr=self.num_steps // self.sample_length
        vs=self._md_cvfe.atom_velocities[:num_corr*self.sample_length] #velocity
        es=self._md_cvfe.atom_energies[:num_corr*self.sample_length] # energies
        eflux = np.zeros((num_corr *self.sample_length,3), dtype="double")
        if self._md_cvfe.fileformat == "l":
            stress_unit = bar * Angstrom ** 3 / EV
        elif self._md_cvfe.fileformat == "x":
            stress_unit = au_force / Bohr2Angstrom ** 2 * self.volume
        else:
            stress_unit = None
        stress = self._md_cvfe.atom_stresses[:num_corr*self.sample_length] * stress_unit
        eflux[:] =  -np.einsum("...ijk,...ij", stress, vs) + np.einsum("...ij,...i", vs, es)
        self.num_corr = num_corr
        self.velocities = np.zeros((self.num_atom, num_corr, self.sample_length, 3), dtype="double")
        self.energies = np.zeros((self.num_atom, num_corr, self.sample_length), dtype="double")

        self.eflux = eflux.reshape((num_corr, self.sample_length,3))

        for i in range(self.num_atom):
            self.velocities[i] = vs[:,i].reshape((num_corr, self.sample_length,3))
            self.energies[i] = es[:,i].reshape((num_corr, self.sample_length))


        #velocities belonging to each set of correlation should be normalized to 0
        if self.is_ai:
            print "force each of the segments to have a mean velocity/force of 0"
            ave_velocity=np.average(self.velocities,axis=2)
            for i in np.arange(self.sample_length):
                self.velocities[:,:,i] -= ave_velocity

    def run_kappa(self):
        self.correlation = np.zeros((self.corr_length,3,3),dtype="double")
        try:
            # raise ImportError
            self.run_auto_correlation_c()
        except ImportError:
            self.run_auto_correlation_py()

        if self.is_sm:
            self.gaussion_smoothing()
        if self.is_wac:
            self._write_correlation()
        from scipy import signal
        b, a  = signal.butter(3, 0.01, 'low')
        for i in range(3):
            for j in range(3):
                self.correlation[:,i,j] = signal.filtfilt(b, a, self.correlation[:,i,j])
        time_series = np.arange(self.corr_length) * self.time_step
        integration = np.trapz(self.correlation, time_series, axis=0)
        self.kappa = integration * self.conversion_factor
        self.print_kappa()

        # self.plot()
    def print_kappa(self):
        print "Thermal conductivity (W/m-K) at %.2f K"%self.temperature


        print "%12s" * 6 %("xx", "yy", "zz", "xy", "xz", "yz")
        kappa = (self.kappa + self.kappa.T) / 2.0
        print "%12.4f" * 6 % (kappa[0,0],kappa[1,1],kappa[2,2],kappa[0,1],kappa[0,2],kappa[1,2])
        if self.dirt is None:
            print "isotropic value"
            print "%12.4f" % (np.einsum("ii",self.kappa) / 3)
        else:
            print "kappa along the wanted direction"
            print self.kappa[self.dirt[0], self.dirt[1]]
        print

    def run_auto_correlation_py(self):
        print "running auto correlation in python language..."
        self.heat_flux_auto_correlation()
            
    def run_auto_correlation_c(self):
        print "running auto correlation in C language..."
        import realmd._ac as ac
        ac.auto_correlation(self.correlation, self.eflux.copy(), 3)
        # ac.auto_correlation(self.correlation, self.velocities.sum(axis=0))

    def heat_flux_auto_correlation(self):
        v=self.eflux
        for j in np.arange(self.num_corr):
            for k in np.arange(self.corr_length):
                cor_time=self.sample_length-k
                for m in np.arange(cor_time):
                    self.correlation[k] += np.outer(v[j, m], v[j, m + k]) / cor_time
        self.correlation /= self.num_corr

    def gaussion_smoothing(self,sigma=None):
        """Apply gaussian smoothing, correlation[num_type][corr_length][3]"""
        print("## Gaussian smoothing...");
        corr_length = self.corr_length
        if sigma is None:
            corr = self.correlation
            oscillation=np.max(np.abs(corr[:-1]-corr[1:]),axis=0)
            peak=np.max(np.abs(corr),axis=0)
            oscillation /= peak
            sigma= corr_length/(5.0*oscillation*len(corr)*self.smooth_tune) # 15.0 has been tuned for many times 
            print "sigma:"
            print sigma
        for i in np.arange(corr_length):
            self.correlation[i] *= exp(-i*i/(2*sigma*sigma))/(sigma*sqrt(2*pi))

    def fourier_transform(self):
        print "#"*75
        print "fourier transform for kappa calculation"
        N=self.corr_length
        delta_t=self.time_step
        delta_f=1/(delta_t*N) # delta frequency unit in THz
        correlation=np.fft.fft(self.correlation,axis=0)[:N/2] / (N/2) * self.time_step# over the first axis(correlation length)
        #correlation_2= np.abs(correlation * correlation.conj())
        frequency=np.arange(N/2)*delta_f
        self.kappa_omega.setdefault("frequency", frequency)
        kappa = np.abs(correlation)  * self.conversion_factor * THz
        self.kappa_omega.setdefault("kappa", kappa)
        self.kappa_omega.setdefault("kappa_cum", np.cumsum(kappa, axis=0) * delta_f)
        
    
    def _write_correlation(self,filename="auto_correlation_kappa.csv"):
        print "#"*75
        print "writing auto correlation data to file %s" %filename
        o=file(filename,'wb')
        csvo=csv.writer(o)
        num_species=self.num_species
        csvo.writerow(["time step"]+
            ["auto correlation(x)"] +
            ["auto correlation(y)"] +
            ["auto correlation(z)"])
        for i in np.arange(self.corr_length):
            flat=self.correlation[i].dot(np.eye(3)).flatten() # choose the diagonal elements and flatten
            csvo.writerow([i]+list(flat))
        del csvo
        o.close()

    
    def plot_freq(self,is_plot=True, is_save=False):
        dirt=self.dirt
        try:
            import matplotlib.pyplot as plt
            freq=self.kappa_omega["frequency"]
            kappa=self.kappa_omega["kappa"]
            kappa_cum = self.kappa_omega['kappa_cum']
            plt.figure(figsize=(16,12))
            if dirt is not None:
                plt.plot(freq, kappa[:,dirt[0],dirt[1]],linewidth=3)
            else:
                plt.plot(freq,np.einsum("ijj",kappa), linewidth=3)
            plt.xlabel("Frequency(THz)",fontsize=18)
            plt.ylabel("Kappa(W/(m K THz))",fontsize=18)
            plt.title("thermal conductivity with frequency", fontsize=24)
            plt.figure(figsize=(16,12))
            if dirt is not None:
                plt.plot(freq, kappa_cum[:,dirt[0],dirt[1]],linewidth=3)
            else:
                plt.plot(freq,np.einsum("ijj",kappa_cum),linewidth=3)
            plt.xlabel("Frequency(THz)",fontsize=18)
            plt.ylabel("Kappa(W/(m K))",fontsize=18)

            if is_save:
                plt.savefig("kappa.pdf")
            if is_plot:
                plt.show()
        except ImportError:
            warning("lib matplotlib is not implemented, continuing anyway...")

    def plot_time(self,is_plot=True, is_save=False):
        try:
            import matplotlib.pyplot as plt
            time_series = np.arange(self.corr_length) * self.time_step
            if self.dirt is not None:
                correlation = self.correlation[:,self.dirt[0], self.dirt[1]]
            else:
                correlation = np.einsum("ijj",self.correlation) / 3.0
            plt.figure(figsize=(8,6))
            eflux = self.eflux[0]
            plt.plot(np.arange(len(self.eflux[0])) * self.time_step, eflux)
            plt.figure(figsize=(8,6))
            plt.plot(time_series[:], correlation / correlation[0])
            # plt.plot(time_series[:], np.cumsum(correlation / correlation[0])/np.arange(1,len(correlation)+1))
            plt.xlabel("Time(ps)",fontsize=18)
            plt.ylabel("Autocorrelation (normalized)",fontsize=18)
            plt.title("Autocorrelation fuction with simulation time", fontsize=24)
            if is_save:
                plt.savefig("ACF.pdf")
            plt.figure(figsize=(8,6))
            kappa_cum = np.cumsum(correlation * self.time_step) * self.conversion_factor
            plt.plot(time_series,kappa_cum)
            plt.xlabel("Time(ps)",fontsize=18)
            plt.ylabel("Thermal conductivity (W/m-K)",fontsize=18)
            plt.title("Thermal conductivity with time", fontsize=24)

            if is_save:
                plt.savefig("KAPPA.pdf")
            if is_plot:
                plt.show()
        except ImportError:
            warning("lib matplotlib is not implemented, continuing anyway...")

class Kappa_HF(Kappa):
    def __init__(self,
                 heat_flux,
                 step_indices=None,
                 temperature=300,
                 volume = None,
                 # atom_types=None,
                 is_ai=False, # is average input
                 sample_l=500, # sample length for correlation
                 corr_l=400, # correlation length
                 smooth_tune=1.0,
                 time_step=1.0, # time step in md calculation(fs)
                 is_diff = False,
                 is_sum=True,# is average in the 3 dimensions
                 is_wac=False,
                 is_sm = False,
                 is_normalize=False,
                 dirt=None,
                 out_file_name="dos.csv"):
        self.kappa_omega={}
        self.num_atom = None

        self.temperature = temperature
        if volume is not None:
            self.volume = volume
        else:
            error("The volume is not set!")
        self.time_step=(step_indices[1]-step_indices[0]) * time_step * fs2ps # [ps]
        self.is_wac=is_wac
        self.is_diff = is_diff
        self.is_normalize=is_normalize
        self.is_ai=is_ai
        self.is_summation=is_sum
        self.num_dim = 3 if is_sum==False else 1
        self.is_sm = is_sm
        self.smooth_tune = smooth_tune
        self.dos_file_name = out_file_name
        self.num_steps = len(step_indices)
        self.sample_length = sample_l
        self.corr_length=corr_l
        self.dirt = dirt
        self.kappa=None
        num_corr=self.num_steps // self.sample_length
        self.eflux = heat_flux[:num_corr*self.sample_length].reshape(num_corr, self.sample_length, 3)
        self.conversion_factor = 1 / self.volume / kb / self.temperature ** 2 * unit_conversion
        self.check_settting()
        self.num_corr=self.num_steps // self.sample_length
        self.print_information()