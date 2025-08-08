import matplotlib.pyplot as plt
import numpy as np
from IsoFATE.isofate.isofunks import *
import IsoFATE.isofate.constants as const

def isoplot(sol,n_atmodeller,Mp,f_atm,Fp,T,M_star,d):

    t_a = sol['time']
    rp_a = sol['Rp']
    # renv_a = sol['renv']
    menv_a = sol['Matm']
    vpot_a = sol['Vpot']
    fenv_a = sol['fatm']
    mloss_a = sol['Mloss']
    phi_a = sol['phi']
    phic_a = sol['phic']
    NH_a = sol['N_H']
    NHe_a = sol['N_He']
    ND_a = sol['N_D']
    NO_a = sol['N_O']
    NC_a = sol['N_C']
    x1_a = sol['x1']
    x2_a = sol['x2']
    PhiH_a = sol['Phi_H']
    PhiHe_a = sol['Phi_He']
    PhiD_a = sol['Phi_D']
    PhiO_a = sol['Phi_O']
    PhiC_a = sol['Phi_C']
    Ts_analytic = sol['T_surf_analytic']
    Ts_atmod = sol['T_surf_atmod']

    if n_atmodeller != 0:
        n_H2O = sol['atmodeller_final']['H2O_atm']
        n_H2 = sol['atmodeller_final']['H2_atm']
        n_O2 = sol['atmodeller_final']['O2_atm']
        n_CO2 = sol['atmodeller_final']['CO2_atm']
        n_CO = sol['atmodeller_final']['CO_atm']
        n_CH4 = sol['atmodeller_final']['CH4_atm']
        n_total_atm = n_H2O + n_H2 + n_O2 + n_CO2 + n_CO + n_CH4 + NHe_a[-1]/const.avogadro
        x_CO2 = n_CO2/n_total_atm
        x_CO = n_CO/n_total_atm
        x_CH4 = n_CH4/n_total_atm
        x_H2 = n_H2/n_total_atm
        x_O2 = n_O2/n_total_atm
        x_H2O = n_H2O/n_total_atm

        N_tot = const.N_H + const.N_He + const.N_D + const.N_O + const.N_C
        N_tot_molecular = (n_H2O + n_H2 + n_O2 + n_CO + n_CO2 + n_CH4)*const.avogadro + NHe_a[-1]
        x_H = const.N_H/N_tot
        x_He = const.N_He/N_tot
        x_He_molecular = NHe_a[-1]/N_tot_molecular
        x_D = const.N_D/N_tot
        x_O = const.N_O/N_tot
        x_C = const.N_C/N_tot
        # print('x_H2O =', x_H2O)
        # print('x_H2 =', x_H2)
        # print('x_He =', x_He_molecular)
        # print('x_O2 =', x_O2)
        # print('x_CO2 =', x_CO2)
        # print('x_CO =', x_CO)
        # print('x_CH4 =', x_CH4)
    # calculate mass fraction of He

    Y = NHe_a*const.mu_He/(NH_a*const.mu_H + NHe_a*const.mu_He + ND_a*const.mu_D + NO_a*const.mu_O + NC_a*const.mu_C)

    # print('\n')
    # print('final f_atm =', fenv_a[-1])
    # print('final f_atm by species =', (NH_a[-1]*const.mu_H + NHe_a[-1]*const.mu_He + ND_a[-1]*const.mu_D + NO_a[-1]*const.mu_O + NC_a[-1]*const.mu_C)/Mp)
    # print('initial f_atm =', fenv_a[0])
    # print('initial f_atm by species =', (N_H*const.mu_H + N_He*const.mu_He + N_D*const.mu_D + N_O*const.mu_O + N_C*const.mu_C)/Mp)
    # print('final D/H =', ND_a[-1]/NH_a[-1]/DtoH_solar, '[Solar]')
    # print('final O/H =', NO_a[-1]/NH_a[-1]/OtoH_protosolar, '[Solar]')
    # print('final X_He (molar concn) =', NHe_a[-1]/(NH_a[-1] + NHe_a[-1] + ND_a[-1] + NO_a[-1] + NC_a[-1])) # molar concentration
    # print('final Y_He (mass concn) =', Y[-1]) # mass fraction
    # print('final planetary radius =', rp_a[-1]/const.Re, 'R_Earth')
    # print('\n')

    # more planetary properties for analytics
    mu = const.mu_solar
    t0 = 1e6/const.s2yr
    r_core = R_core(Mp)
    r_env = R_env(Mp, f_atm, Fp, t0)
    r_atm = R_atm(T, Mp, r_core, r_env, mu)
    Rp = r_core + r_env + r_atm
    # Rp = r_core
    R_B = R_Bondi(Mp, mu, T) # Bondi radius [m]
    R_H = R_Hill(Mp, M_star, d) # Hill radius [m]
    # Rp = r_core # use this if rad_evol = False for analytics to match isofate
    # Rp = Rp_override
    Rp = np.min([Rp, R_B, R_H]) # [m]
    # R_avg = (Rp + r_core)/2
    # A_avg = 4*np.pi*R_avg**2
    # Matm_avg = f_atm*Mp/2
    # phi_avg = Matm_avg/A_avg/(time/2)
    A = 4*np.pi*Rp**2 # [m2]
    # g = G*Mp/Rp**2
    M_atm = Mp*f_atm # [kg]

    plt.rcParams["figure.figsize"] = (14,8)
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'axes.titlesize': 11})

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex = False)
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex = False) # use for just phi, x2, and N1/N2 plots
    plt.subplots_adjust(wspace = 0.3)

    # phi
    g = G*Mp/rp_a**2
    H_H = const.R_gas*T/(const.M_H*g) # D scale height [m]
    H_D = const.R_gas*T/(const.M_D*g) # D scale height [m]
    ax1.plot(t_a*const.s2yr, PhiH_a*const.mu_H, color = 'black', label = 'H flux')
    ax1.plot(t_a*const.s2yr, PhiHe_a*const.mu_He, color = 'grey', label = 'He flux')
    ax1.plot(t_a*const.s2yr, phic_a, '--', color = 'grey', label = 'He critical')
    if ND_a[0] != 0:
        ax1.plot(t_a*const.s2yr, PhiD_a*const.mu_D, color = 'orangered', label = 'D flux')
        ax1.plot(t_a*const.s2yr, const.b_H_D(T)*x1_a*(const.mu_D - const.mu_H)/H_H, '--', color = 'orangered', label = 'D critical') # D/H critical flux
    if NO_a[0] != 0:
        ax1.plot(t_a*const.s2yr, PhiO_a*const.mu_O, color = 'green', label = 'O flux')
        ax1.plot(t_a*const.s2yr, const.b_H_O(T)*x1_a*(const.mu_O - const.mu_H)/H_H, '--', color = 'green', label = 'O critical') # O/H critical flux
    if NC_a[0] != 0:
        ax1.plot(t_a*const.s2yr, PhiC_a*const.mu_C, color = 'gold', label = 'C flux')
        ax1.plot(t_a*const.s2yr, const.b_H_C(T)*x1_a*(const.mu_C - const.mu_H)/H_H, '--', color = 'gold', label = 'C critical') # C/H critical flux
    ax1.plot(t_a*const.s2yr, phi_a, ':', color = 'mediumslateblue', label = 'total flux')
    cross = np.where(np.abs(phi_a - phic_a) < 1e-12)
    if len(cross[0]) != 0:
        ax1.axvline(t_a[cross][0]*const.s2yr, ls = '--', lw = 0.5, color = 'coral')
        print('critical flux at ', t_a[cross][0]*const.s2yr/1e9, 'Gyr')
    ax1.set_ylabel('phi [kg m$^{-2}$ s$^{-1}$]', labelpad = 2)
    ax1.legend(fontsize = 8.5, frameon = False, framealpha = 0.7, loc = 3, ncol = 3)
    ax1.set_ylim(PhiH_a[0]*const.mu_H/1e8, PhiH_a[0]*const.mu_H*100)
    ax1.set_yscale('log')

    # D/H, O/H
    ax2.loglog(t_a*const.s2yr, x2_a, color = 'maroon', label = 'IsoFATE')
    ax2.set_ylabel('N$_{He}$/(N$_H$+N$_{He}$)', labelpad = 2)
    # if ND_a[0] != 0:
    #     ax2.loglog(t_a*const.s2yr, ND_a/NH_a, color = 'maroon', label = 'IsoFATE')
    #     ax2.set_ylabel('D/H [Solar]', labelpad = 2)
    # if NO_a[0] != 0:
    #     ax2.loglog(t_a*const.s2yr, NO_a/NH_a, color = 'maroon', label = 'IsoFATE')
    #     ax2.set_ylabel('O/H [Solar]', labelpad = 2)

    # ax2.set_title('N$_2$/N$_1$ [xSolar]')
    # ax2.set_xlabel('Time [yr]')
    # if mechanism == 'fix phi subcritical':
    #     x2_analytic = x2sub(x2_0, tau, t_a*const.s2yr)
    #     # x2_analytic = x2sub2(x2_0, N1_0, A, Phi1, t_a*const.s2yr)
    #     ax2.loglog(t_a*const.s2yr, x2_analytic/dh_a[0,0], '--', color = 'deeppink', zorder = 10, label = 'analytic')
    #     ax2.legend(shadow = True)
    # elif mechanism == 'fix phi supercritical':
    #     x2_analytic = x2super(x2_0, Mp, Rp, T, Phi, tau/const.s2yr, t_a)
    #     ax2.loglog(t_a*const.s2yr, x2_analytic/dh_a[0,0], '--', color = 'deeppink', zorder = 10, label = 'analytic')
    #     ax2.legend(shadow = True)
    # else:
        # x2_analytic = analytic_soln(x2_0, N1_0, const.mu1, const.mu2, x1_a[0], x2_a[0], Mp, rp_a[0]*const.Re, fenv_a[0], T, phi_a[0], phic_a[0], t_a)
        # ax2.loglog(t_a*const.s2yr, x2_analytic[0]/dh_a[0,0], '--', color = 'deeppink', zorder = 10, label = 'analytic')
        # ax2.legend(shadow = True)
    # ax2.set_title('atmospheric fractionation')
    if len(cross[0]) != 0:
        ax1.axvline(t_a[cross][0]*const.s2yr, ls = '--', lw = 0.5, color = 'coral')
    #ax2.set_xlabel('time [yr]')

    # N_x
    ax3.plot(t_a*const.s2yr, NH_a/const.avogadro, color = 'black', label = 'N$_{H}$')
    # ax3.set_xlabel('Time [yr]')
    if len(cross[0]) != 0:
        ax3.axvline(t_a[cross][0]*const.s2yr, ls = '--', lw = 0.5, color = 'coral')
    if NHe_a[0] != 0:
        ax3.plot(t_a*const.s2yr, NHe_a/const.avogadro, color = 'grey', label = 'N$_{He}$')
    if ND_a[0] != 0:
        ax3.plot(t_a*const.s2yr, ND_a/const.avogadro, color = 'orangered', label = 'N$_D$')
    if NO_a[0] != 0:
        ax3.plot(t_a*const.s2yr, NO_a/const.avogadro, color = 'green', label = 'N$_{O}$')
    if NC_a[0] != 0:
        ax3.plot(t_a*const.s2yr, NC_a/const.avogadro, color = 'gold', label = 'N$_{C}$')
    # if mechanism == 'fix phi subcritical' or mechanism == 'fix phi supercritical':
    #     ax3.loglog(t_a*const.s2yr, N1(t_a, A, Phi, N1_0)/const.avogadro, '--', color = 'skyblue', zorder = 10, label = 'analytic N$_{H}$') # changed H_0 to N1_0
    #     ax3.legend(shadow = True)
    #else:
        #ax3.loglog(t_a*const.s2yr, N1(t_a, A, phi_c/const.mu1, N1_0), '--', color = 'deeppink', zorder = 10, label = 'analytic') # changed H_0 to N1_0
        # ax3.loglog(t_a*const.s2yr, N1_mod(t_a, 4*np.pi*r_core**2, phi_c/const.mu1, N1_0), '--', color = 'deeppink', zorder = 10, label = 'analytic') # changed H_0 to N1_0
    # ax3.set_title('hydrogen number')
    ax3.set_ylabel('atmospheric moles', labelpad = 2)
    ax3.legend(frameon = False, fontsize = 9)
    ax3.set_ylim(NH_a[0]/const.avogadro/1e12, NH_a[0]/const.avogadro*100)
    ax3.set_yscale('log')
    #ax3.set_xlabel('time [yr]')

    # planet radius
    ax4.loglog(t_a*const.s2yr, rp_a/const.Re, color = 'mediumseagreen')
    # ax4.set_title('planetary radius')
    ax4.set_ylabel('radius [R$_\oplus$]', labelpad = 2)
    #ax4.set_xlabel('time [yr]')

    # envelope radius
    # ax5.loglog(t_a*const.s2yr, renv_a[0]*1000, color = 'skyblue')
    # # ax5.set_title('envelope radius')
    # ax5.set_ylabel('R$_{env}$ [km]', labelpad = 2)
    # #ax5.set_xlabel('time [yr]')

    ax5.plot(t_a*const.s2yr, Ts_atmod, color = 'mediumslateblue')
    ax5.set_xscale('log')
    ax5.set_ylabel('surface temp [K]', labelpad = 2)
    ax5.set_ylim(-100, 6100)
    ax5.annotate(f'Rp = {round(rp_a[-1]/const.Re, 2)} const.Re, Mp = {round(Mp/const.Me, 2)} const.Me, Teq = {round(T, 0)} K', (1e6, 5500), fontsize = 10)
    ax5.annotate('final fatm:'+str(round(fenv_a[-1], 6)), (1e6, 5000), fontsize = 10)
    ax5.annotate('final D/H:'+str(round(ND_a[-1]/NH_a[-1]/const.DtoH_solar, 2))+' [Solar]', (1e6, 4500), fontsize = 10)
    if n_atmodeller != 0:
        ax5.annotate('final He concn:'+str(round(x_He_molecular, 4))+'='+str(round(Y[-1], 4))+' kg/kg', (1e6, 4000), fontsize = 10)
        ax5.annotate('final O2 concn:'+str(round(x_O2, 4)), (1e6, 3500), fontsize = 10)
        ax5.annotate('final H2O concn:'+str(round(x_H2O, 4)), (1e6, 3000), fontsize = 10)
        ax5.annotate('final CO2 concn:'+str(round(x_CO2, 4)), (1e6, 2500), fontsize = 10)
        ax5.annotate('final CO concn:'+str(round(x_CO, 4)), (1e6, 2000), fontsize = 10)
        ax5.annotate('final CH4 concn:'+str(round(x_CH4, 4)), (1e6, 1500), fontsize = 10)
        ax5.annotate('final H2 concn:'+str(round(x_H2, 4)), (1e6, 1000), fontsize = 10)

    # envelope mass
    ax6.loglog(t_a*const.s2yr, menv_a/const.Me, color = 'midnightblue')
    # ax6.set_title('envelope mass')
    ax6.set_ylabel('M$_{env}$ [M$_\oplus$]', labelpad = 2)
    #ax6.set_xlabel('time [yr]')

    # grav potential
    ax7.loglog(t_a*const.s2yr, vpot_a, color = 'orange')
    # ax7.set_title('gravitational potential')
    ax7.set_ylabel('V$_{pot}$ [J/kg]', labelpad = 2)
    ax7.set_xlabel('time [yr]')

    # mass loss
    ax8.loglog(t_a*const.s2yr, mloss_a, color = 'crimson')
    # ax8.set_title('mass loss per time step')
    ax8.set_ylabel('$\Delta$ mass [kg]', labelpad = 2)
    ax8.set_xlabel('time [yr]')

    # f_env or system parameters
    ax9.set_xlabel('time [yr]')

    ### model parameters

    # if adaptive_steps == False:
    # ax9.annotate('f_atm ='+str(f_atm)+',', (1e8, 0.9))
    # ax9.annotate('comp:'+str(species), (3e9, 0.9))
    # ax9.annotate('Mp ='+str(round(Mp/const.Me, 2))+' M$_\oplus$', (1e8, 0.78))
    # ax9.annotate('Rp(t=0) ='+str(round(Rp/const.Re, 2))+' R$_\oplus$', (1e8, 0.66))
    # ax9.annotate('F$_{XUV}$(t=0) ='+str(round(F0, 2))+' W/m2', (1e8, 0.54))
    # ax9.annotate('P ='+str(round(P*s2day, 2))+' days', (1e8, 0.42))
    # ax9.annotate('Fp ='+str(round(Fp, 2))+' W/m2', (1e8, 0.30))
    # ax9.annotate('Teq ='+str(round(T, 2))+' K', (1e8, 0.18))
    # ax9.annotate('mechanism ='+str(mechanism), (1e8, 0.06))
    # ax9.tick_params(left = 'False')
    # ax9.set_yticks([])

    ### atmospheric mass fraction

    ax9.set_ylabel('f_env [%]', labelpad = 2)
    ax9.plot(t_a*const.s2yr, fenv_a*100, color = 'gold')
    ax9.set_xscale('log')

    plt.tight_layout()

    # plt.savefig('/Users/collin/Documents/Harvard/const.Research/atm_escape/IsoFATE/case_studies/LHS1140b_f0017_Fi1_tjump_Ffinal170_XUV+RR_atmod1e2_ntime1e5_t10e9_tpms3e8_v2.png', dpi = 300, bbox_inches = 'tight')

    plt.show();
