import uproot as up
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vector
import zfit
import mplhep as hep
import yaml
import seaborn as sns
import particle
import awkward
from CMS_cuts import CMS_cut_func
import sys
from uncertainties import ufloat
#hep.style.use('ATLAS')

def analysis(input_file,det_output_csv,det_output_plot,part_output_plot,part_output_csv):
    tree=up.open(input_file+':Delphes')
 
    base_features=['_size','.PT','.Eta','.Phi']
    particles=['Muon','Electron','Photon']
    neutrino_features=['MissingET.MET','MissingET.Eta','MissingET.Phi','MissingET_size']+['Jet_size','Jet.PT','Jet.Eta','Event.Weight','Electron.Charge','Muon.Charge','Jet.Phi']
    event_features=[i+j for i in particles for j in base_features]+neutrino_features
    events_df=tree.arrays(event_features,library="pd")

    

    FILE_SIZE=len(events_df)
    events_df_full=events_df.copy()

    events_df=events_df.loc[ ((events_df['Photon_size']>0)& (events_df['MissingET_size']==1) & ((events_df['Muon_size']==1)) & ( events_df['Electron_size']==0 ) ) |
                            ( (events_df['Photon_size']>0)&(events_df['MissingET_size']==1) & ((events_df['Electron_size']==1))   &( events_df['Muon_size']==0) )  ]
    def extract_single_element(x):
            if isinstance(x, list) or isinstance(x,np.ndarray) or isinstance(x,awkward.highlevel.Array):
                if len(x) == 1:
                    return x[0]
                elif len(x)==0:
                    return np.nan
                elif len(x)>1:
                    return np.array(x)
            elif isinstance(x, (np.float32, np.float64, float,int)): 
                return x
            else:
                return np.nan  # If
    events_df = events_df.map(extract_single_element)
    for i in events_df.index:
    #print(events_df.loc[i,'Muon_size'])
        if events_df.loc[i,'Muon_size']==1:
            lepton='Muon'
            pdg=13
        else:
            lepton='Electron'
            pdg=11
        
        eta_l=events_df.loc[i,lepton+'.Eta']
        phi_l=events_df.loc[i,lepton+'.Phi']
        pt_l=events_df.loc[i,lepton+'.PT']
        dressed_lepton=vector.obj(eta=eta_l,phi=phi_l,pt=pt_l,mass=particle.Particle.from_pdgid(pdg).mass/1000)
        

        if events_df.loc[i,'Photon_size']>1:
            candidate_photons=np.array(range(events_df.loc[i,'Photon_size']))
            for j in range(events_df.loc[i,'Photon_size']):
                temp=np.sqrt((eta_l-events_df.loc[i,'Photon.Eta'][j])**2+(phi_l-events_df.loc[i,'Photon.Phi'][j])**2)

                if temp<0.1:
                    y=vector.obj(eta=events_df.loc[i,'Photon.Eta'][j],phi=events_df.loc[i,'Photon.Phi'][j],pt=events_df.loc[i,'Photon.PT'][j],mass=0)
                    dressed_lepton=dressed_lepton+y
                    candidate_photons=np.delete(candidate_photons,j)
            for j in candidate_photons:
                temp=np.sqrt((eta_l-events_df.loc[i,'Photon.Eta'][j])**2+(phi_l-events_df.loc[i,'Photon.Phi'][j])**2)
                if temp<0.4:
                    candidate_photons=np.delete(candidate_photons,j)
            if len(candidate_photons)>0:
                pt_temp=events_df_full.loc[i,'Photon.PT'][candidate_photons]

                leading_photon=np.argmax(pt_temp)
                
                events_df.at[i,'Photon_size']=1
                events_df.at[i,'Photon.PT']=events_df_full.loc[i,'Photon.PT'][leading_photon]
                events_df.at[i,'Photon.Eta']=events_df_full.loc[i,'Photon.Eta'][leading_photon]
                events_df.at[i,'Photon.Phi']=events_df_full.loc[i,'Photon.Phi'][leading_photon]
            else:
                events_df.at[i,'Photon_size']=0


        else:
            temp=np.sqrt((eta_l-events_df.loc[i,'Photon.Eta'])**2+(phi_l-events_df.loc[i,'Photon.Phi'])**2)

            if temp<0.1:
                y=vector.obj(eta=events_df.loc[i,'Photon.Eta'],phi=events_df.loc[i,'Photon.Phi'],pt=events_df.loc[i,'Photon.PT'],mass=0)
                dressed_lepton=dressed_lepton+y
                events_df.at[i,'Photon_size']=0
        
            elif temp<0.4:
                events_df.at[i,'Photon_size']=0
            else:
                events_df.at[i,'Photon_size']=1
        
        #print([dressed_lepton.pt])
        events_df.at[i,lepton+'.PT']=dressed_lepton.pt
        events_df.at[i,lepton+'.Eta']=dressed_lepton.eta
        events_df.at[i,lepton+'.Phi']=dressed_lepton.phi


    events_df = events_df.map(extract_single_element)









    CMS_cut=events_df.loc[(events_df['MissingET.MET']>40) &(events_df['Photon.PT']>30) & (np.abs(events_df['Photon.Eta'])<2.5)&
            
            ( (events_df['Electron.PT']>35) | (events_df['Muon.PT']>30) ) 
            &
            (  (    (np.abs(events_df['Electron.Eta'])<1.44) | (np.abs(events_df['Electron.Eta'])>1.57)      &  
                    (np.abs(events_df['Electron.Eta'])<2.5)    )          | (np.abs(events_df['Muon.Eta'])<2.4)  )  
            
            &(    (  np.sqrt((events_df['Electron.Eta'] - events_df['Photon.Eta'])**2 +(events_df['Electron.Phi'] -events_df['Photon.Phi'] )**2  )>0.7
                    )  |
                    ( np.sqrt((events_df['Muon.Eta'] - events_df['Photon.Eta'])**2 +(events_df['Muon.Phi'] -events_df['Photon.Phi'] )**2  )>0.7)       )
            ].copy()
    CMS_cut['lepton_mass']=0.0
    CMS_cut.loc[CMS_cut['Muon_size']==1,'lepton_mass']=particle.Particle.from_pdgid(13).mass/1000
    CMS_cut.loc[CMS_cut['Electron_size']==1,'lepton_mass']=particle.Particle.from_pdgid(11).mass/1000

    CMS_cut['M_ly']=0.0

    CMS_cut.loc[CMS_cut['Muon_size']==1,'M_ly']=np.sqrt(       
        (CMS_cut['Photon.PT']*np.cosh(CMS_cut['Photon.Eta'])  + np.sqrt(   (CMS_cut['Muon.PT']*np.cosh(CMS_cut['Muon.Eta']))**2+CMS_cut['lepton_mass']**2    )  )**2  
            - (   CMS_cut['Muon.PT']*np.cos(CMS_cut['Muon.Phi']) +CMS_cut['Photon.PT']*np.cos(CMS_cut['Photon.Phi'])  )**2
            - (   CMS_cut['Muon.PT']*np.sin(CMS_cut['Muon.Phi']) +CMS_cut['Photon.PT']*np.sin(CMS_cut['Photon.Phi'])  )**2
            - (   CMS_cut['Muon.PT']*np.sinh(CMS_cut['Muon.Eta']) +CMS_cut['Photon.PT']*np.sinh(CMS_cut['Photon.Eta'])  )**2
                )
    CMS_cut.loc[CMS_cut['Electron_size']==1,'M_ly']=np.sqrt(       
        (CMS_cut['Photon.PT']*np.cosh(CMS_cut['Photon.Eta'])  + np.sqrt(   (CMS_cut['Electron.PT']*np.cosh(CMS_cut['Electron.Eta']))**2+CMS_cut['lepton_mass']**2    )  )**2  
            - (   CMS_cut['Electron.PT']*np.cos(CMS_cut['Electron.Phi']) +CMS_cut['Photon.PT']*np.cos(CMS_cut['Photon.Phi'])  )**2
            - (   CMS_cut['Electron.PT']*np.sin(CMS_cut['Electron.Phi']) +CMS_cut['Photon.PT']*np.sin(CMS_cut['Photon.Phi'])  )**2
            - (   CMS_cut['Electron.PT']*np.sinh(CMS_cut['Electron.Eta']) +CMS_cut['Photon.PT']*np.sinh(CMS_cut['Photon.Eta'])  )**2
                )


    CMS_cut=CMS_cut.loc[   (((CMS_cut['M_ly']<70)|(CMS_cut['M_ly']>110)  ) & (CMS_cut['Electron_size']==1)) |
                    (((CMS_cut['M_ly']<70)|(CMS_cut['M_ly']>100)  ) & (CMS_cut['Muon_size']==1))
    ].copy()

    CMS_cut['Mt_lv']=0.0


    CMS_cut.loc[CMS_cut['Electron_size']==1,'Mt_lv']=np.sqrt(  
        ( CMS_cut['MissingET.MET']+ np.sqrt((CMS_cut['Electron.PT'])**2+CMS_cut['lepton_mass']**2))**2 
        - (   CMS_cut['Electron.PT']*np.cos(CMS_cut['Electron.Phi']) +CMS_cut['MissingET.MET']*np.cos(CMS_cut['MissingET.Phi'])  )**2
            - (   CMS_cut['Electron.PT']*np.sin(CMS_cut['Electron.Phi']) +CMS_cut['MissingET.MET']*np.sin(CMS_cut['MissingET.Phi'])  )**2   )

    CMS_cut.loc[CMS_cut['Muon_size']==1,'Mt_lv']=np.sqrt(  
        ( CMS_cut['MissingET.MET']+ np.sqrt((CMS_cut['Muon.PT'])**2+CMS_cut['lepton_mass']**2))**2 
        - (   CMS_cut['Muon.PT']*np.cos(CMS_cut['Muon.Phi']) +CMS_cut['MissingET.MET']*np.cos(CMS_cut['MissingET.Phi'])  )**2
            - (   CMS_cut['Muon.PT']*np.sin(CMS_cut['Muon.Phi']) +CMS_cut['MissingET.MET']*np.sin(CMS_cut['MissingET.Phi'])  )**2   )


    for i in CMS_cut.index:
        n_jet=CMS_cut['Jet_size'][i]
        if n_jet==0:
            continue
        else:
            jet_pt=np.array(CMS_cut['Jet.PT'][i])
            jet_eta=np.array(CMS_cut['Jet.Eta'][i])
            jet_phi=np.array(CMS_cut['Jet.Phi'][i])

            pt_veto=np.nonzero(jet_pt<30)[0]
            eta_veto=np.nonzero(np.abs(jet_eta)<2.5)[0]

            jet_veto=np.intersect1d(pt_veto,eta_veto)

            if len(jet_veto>0):
                if n_jet-len(jet_veto)==0:
                    CMS_cut.loc[i,'Jet.PT']=np.nan
                    CMS_cut.loc[i,'Jet.Eta']=np.nan
                    CMS_cut.loc[i,'Jet.Phi']=np.nan
                else:

                    CMS_cut.at[i,'Jet.PT']=np.delete(jet_pt,jet_veto)
                    CMS_cut.at[i,'Jet.Eta']=np.delete(jet_eta,jet_veto)
                    CMS_cut.at[i,'Jet.Phi']=np.delete(jet_phi,jet_veto)
                    CMS_cut.loc[i,'Jet_size']=n_jet-len(jet_veto)
                
        
        

    CMS_cut['Mt_cluster']=0.0

    CMS_cut.loc[CMS_cut['Electron_size']==1,'Mt_cluster']=np.sqrt(  (np.sqrt( CMS_cut['M_ly']**2  +
                                                                            (CMS_cut['Electron.PT']*np.cos(CMS_cut['Electron.Phi']) +CMS_cut['Photon.PT']*np.cos(CMS_cut['Photon.Phi'])  )**2 + 
                                                                            (CMS_cut['Electron.PT']*np.sin(CMS_cut['Electron.Phi']) +CMS_cut['Photon.PT']*np.sin(CMS_cut['Photon.Phi'])  )**2    )+ 
                                                                            CMS_cut['MissingET.MET']  )**2  -  
                                                                            (CMS_cut['Electron.PT']*np.cos(CMS_cut['Electron.Phi']) +CMS_cut['Photon.PT']*np.cos(CMS_cut['Photon.Phi']) +CMS_cut['MissingET.MET']*np.cos(CMS_cut['MissingET.Phi']) )**2 - 
                                                                                (CMS_cut['Electron.PT']*np.sin(CMS_cut['Electron.Phi']) +CMS_cut['Photon.PT']*np.sin(CMS_cut['Photon.Phi']) +CMS_cut['MissingET.MET']*np.sin(CMS_cut['MissingET.Phi']))**2  )




    CMS_cut.loc[CMS_cut['Muon_size']==1,'Mt_cluster']=np.sqrt(  (np.sqrt( CMS_cut['M_ly']**2  +
                                                                            (CMS_cut['Muon.PT']*np.cos(CMS_cut['Muon.Phi']) +CMS_cut['Photon.PT']*np.cos(CMS_cut['Photon.Phi'])  )**2 + 
                                                                            (CMS_cut['Muon.PT']*np.sin(CMS_cut['Muon.Phi']) +CMS_cut['Photon.PT']*np.sin(CMS_cut['Photon.Phi'])  )**2    )+ 
                                                                            CMS_cut['MissingET.MET']  )**2  -  
                                                                            (CMS_cut['Muon.PT']*np.cos(CMS_cut['Muon.Phi']) +CMS_cut['Photon.PT']*np.cos(CMS_cut['Photon.Phi']) +CMS_cut['MissingET.MET']*np.cos(CMS_cut['MissingET.Phi']) )**2 - 
                                                                                (CMS_cut['Muon.PT']*np.sin(CMS_cut['Muon.Phi']) +CMS_cut['Photon.PT']*np.sin(CMS_cut['Photon.Phi']) +CMS_cut['MissingET.MET']*np.sin(CMS_cut['MissingET.Phi']))**2  )


    cross_section_inclusive=ufloat(0.9919,0.0016)
    e=len(CMS_cut)/FILE_SIZE

    events_df_full = events_df_full.map(extract_single_element)
    W_CONSTANT=138*995.7/np.sum(events_df_full['Event.Weight'])


    epsilon=ufloat(   e, np.sqrt( e*(1-e)/FILE_SIZE  )  )

    cross_section_integrated=cross_section_inclusive*epsilon *1000



    fig,axes=plt.subplots(2,2,figsize=(12,10))


    Pt_lepton=pd.concat([CMS_cut['Muon.PT'],CMS_cut['Electron.PT']],axis=0,ignore_index=True)
    var=Pt_lepton
    weight=pd.concat([CMS_cut['Event.Weight'],CMS_cut['Event.Weight']],axis=0,ignore_index=True)

    counts, bins = np.histogram(var, 25,range=(30, 140),weights=W_CONSTANT*weight)
    bin_centres = (bins[:-1] + bins[1:])/2.
    err = np.sqrt(counts)
    sns.histplot(x=var,bins=25,element='step',binrange=[30,140],edgecolor='black',ax=axes[0][0],weights=W_CONSTANT*weight)
    axes[0][0].errorbar(bin_centres,counts,yerr=err,fmt='o',color='black',markersize=5,elinewidth=2.5,label='Data')

    axes[0][0].set_xlim(30,140)
    axes[0][0].set_ylabel('Counts / 4.4 GeV')
    axes[0][0].set_xlabel(r'$\mathrm{p_{T}^{l}}$'+' (GeV)')




    Pt_photon=CMS_cut['Photon.PT']
    var=Pt_photon

    counts, bins = np.histogram(var, 9,range=(30, 1000),weights=W_CONSTANT*CMS_cut['Event.Weight'])
    bin_centres = (bins[:-1] + bins[1:])/2.
    err = np.sqrt(counts)
    sns.histplot(x=var,bins=9,element='step',edgecolor='black',binrange=[30,1000],ax=axes[0][1],weights=W_CONSTANT*CMS_cut['Event.Weight'])
    axes[0][1].errorbar(bin_centres,counts,yerr=err,fmt='o',color='black',markersize=5,elinewidth=2.5,label='Data')
    axes[0][1].set_yscale('log')
    axes[0][1].set_ylabel('Counts / 12.2 GeV')
    axes[0][1].set_xlabel(r'$\mathrm{p_{T}^{\gamma}}$'+' (GeV)')
    axes[0][1].set_xlim(30,1000)







    var=CMS_cut['Mt_lv']

    counts, bins = np.histogram(var, 30,range=(0, 200),weights=W_CONSTANT*CMS_cut['Event.Weight'])
    bin_centres = (bins[:-1] + bins[1:])/2.
    err = np.sqrt(counts)
    sns.histplot(x=var,bins=30,element='step',edgecolor='black',binrange=[0,200],ax=axes[1][0],weights=W_CONSTANT*CMS_cut['Event.Weight'])
    axes[1][0].errorbar(bin_centres,counts,yerr=err,fmt='o',color='black',markersize=5,elinewidth=2.5,label='Data')
    axes[1][0].set_ylabel('Counts / 6.6 GeV')
    axes[1][0].set_xlabel(r'$\mathrm{m_{T}(l,p_{T}^{miss})}$'+' (GeV)')






    var=CMS_cut['Mt_cluster']

    counts, bins = np.histogram(var, 50,weights=W_CONSTANT*CMS_cut['Event.Weight'])
    bin_centres = (bins[:-1] + bins[1:])/2.
    err = np.sqrt(counts)
    sns.histplot(x=var,bins=50,element='step',edgecolor='black',ax=axes[1][1],weights=W_CONSTANT*CMS_cut['Event.Weight'])
    axes[1][1].errorbar(bin_centres,counts,yerr=err,fmt='o',color='black',markersize=5,elinewidth=2.5,label='Data')
    axes[1][1].set_yscale('log')
    axes[1][1].set_ylabel('Counts / 50 GeV')
    axes[1][1].set_xlabel(r'$\mathrm{m_{T}^{l \gamma \nu}}$'+' (GeV)')

    plt.suptitle('CMS Valdiation Detector level',size=25)


    plt.tight_layout()
    plt.savefig(det_output_plot,dpi=800)
    plt.close()
    CMS_cut.to_csv(det_output_csv)

    ### particle level


    tree=up.open(input_file+':Delphes')


    base_features=['.M1',
                '.PT','.Eta','.Phi',
                '.M2',
                '.D1',
                '.D2',
                '.PID',
                '.E',
                '.Status']
    particles=['Particle']
    event_features=[i+j for i in particles for j in base_features]+['GenJet_size','GenJet.PT','GenJet.Eta','GenJet.Phi','GenMissingET_size','GenMissingET.MET','GenMissingET.Phi','GenMissingET.Eta']
    events_df=tree.arrays(event_features,library="pd")



    events_df[['Nu_y_coord','Charged_lepton','Dressed_lepton.PT','Dressed_lepton.Eta','Dressed_lepton.Phi']]=None




    for i in range(len(events_df)):
        index=range(len(events_df['Particle.PID'][i]))
        p=events_df['Particle.PID'][i]
        m1=events_df['Particle.M1'][i]
        m2=events_df['Particle.M2'][i]
        d1=events_df['Particle.D1'][i]
        d2=events_df['Particle.D2'][i]
        pt=events_df['Particle.PT'][i]
        status=events_df['Particle.Status'][i]
        eta=events_df['Particle.Eta'][i]
        phi=events_df['Particle.Phi'][i]

        charged_lepton=np.where( (np.abs(p)==11)|(np.abs(p)==13) )[0]
        neutrino=np.where( (np.abs(p)==12)|(np.abs(p)==14) )[0]
        status_mask=np.where(status==1)[0]

        photon=np.where(p==22)[0]

        particle_matrix=np.array([index,p,m1,m2,status,d1,d2,np.array(p[m1]),np.array(p[m2]),pt]).T
        
        #particle_matrix=particle_matrix[particle_matrix[:,9].argsort()]


        #charged_lepton_matrix=particle_matrix[(np.abs(particle_matrix[:,1])==11) | (np.abs(particle_matrix[:,1])==13)]
        
        #neutrino_matrix=particle_matrix[(np.abs(particle_matrix[:,1])==12) | (np.abs(particle_matrix[:,1])==14)]

        #photon_matrix=particle_matrix[particle_matrix[:,1]==22 ]

        


        true_photon=[]

        for k in np.intersect1d(photon,status_mask):
            current_row_index = np.where(particle_matrix[:,0]==k)[0]
            #print(current_row_index[0])

            # Get the initial flag from the second value of the starting row
            flag = particle_matrix[current_row_index[0]][1]
            #print(particle_matrix[current_row_index])
            while True:
                # Extract the next row index from the third value of the current row
                next_row_index = int(particle_matrix[current_row_index][0][2])

                # Check the 6th value of the current row (index 5) and compare with the flag
                if (particle_matrix[current_row_index][0][7] != flag):# and (particle_matrix[current_row_index][0][4] != 1):
                    # Output the current row since it breaks the condition
                    #print("Output row:", particle_matrix[current_row_index][0])
                    
                    if (np.abs(particle_matrix[current_row_index][0][7])<37) and (np.abs(particle_matrix[current_row_index][0][7])!=15):
                        true_photon.append(k)

                    #true_mother.append(np.array([particle_matrix[current_row_index][0][7],particle_matrix[current_row_index][0][8]]))
                    break

                # Move to the next row based on the third column
                current_row_index = np.where(particle_matrix[:,0]==next_row_index)[0]
        true_photon=np.array(true_photon)
        

        #print(true_photon[np.argmax(photon_pt)],np.max(photon_pt))


        true_ch_lepton=[]

        for k in np.intersect1d(charged_lepton,status_mask):
            current_row_index = np.where(particle_matrix[:,0]==k)[0]
            #print(current_row_index[0])

            # Get the initial flag from the second value of the starting row
            flag = particle_matrix[current_row_index[0]][1]
            #print(particle_matrix[current_row_index])
            while True:
                # Extract the next row index from the third value of the current row
                next_row_index = int(particle_matrix[current_row_index][0][2])

                # Check the 6th value of the current row (index 5) and compare with the flag
                if (particle_matrix[current_row_index][0][7] != flag):# and (particle_matrix[current_row_index][0][4] != 1):
                    # Output the current row since it breaks the condition
                    #print("Output row:", particle_matrix[current_row_index][0])
                    
                    if (np.abs(particle_matrix[current_row_index][0][7])<37 )and (np.abs(particle_matrix[current_row_index][0][7])!=15):
                        true_ch_lepton.append(k)

                    #true_mother.append(np.array([particle_matrix[current_row_index][0][7],particle_matrix[current_row_index][0][8]]))
                    break

                # Move to the next row based on the third column
                current_row_index = np.where(particle_matrix[:,0]==next_row_index)[0]
        true_ch_lepton=np.array(true_ch_lepton)
        ch_lepton_pt=pt[true_ch_lepton]


        true_neutrino_lepton=[]

        for k in np.intersect1d(neutrino,status_mask):
            current_row_index = np.where(particle_matrix[:,0]==k)[0]
            #print(current_row_index[0])

            # Get the initial flag from the second value of the starting row
            flag = particle_matrix[current_row_index[0]][1]
            #print(particle_matrix[current_row_index])
            while True:
                # Extract the next row index from the third value of the current row
                next_row_index = int(particle_matrix[current_row_index][0][2])

                # Check the 6th value of the current row (index 5) and compare with the flag
                if (particle_matrix[current_row_index][0][7] != flag):# and (particle_matrix[current_row_index][0][4] != 1):
                    # Output the current row since it breaks the condition
                    #print("Output row:", particle_matrix[current_row_index][0])
                    
                    if (np.abs(particle_matrix[current_row_index][0][7])<37) and (np.abs(particle_matrix[current_row_index][0][7])!=15):
                        true_neutrino_lepton.append(k)

                    #true_mother.append(np.array([particle_matrix[current_row_index][0][7],particle_matrix[current_row_index][0][8]]))
                    break

                # Move to the next row based on the third column
                current_row_index = np.where(particle_matrix[:,0]==next_row_index)[0]
            
        true_neutrino_lepton=np.array(true_neutrino_lepton)
        neutrino_pt=pt[true_neutrino_lepton]

        dressed_leptons=[]
        #print(true_ch_lepton)
        for lepton in true_ch_lepton:
            candidate_photons=true_photon
            dressed_lepton=vector.obj(pt=pt[lepton],eta=eta[lepton],phi=phi[lepton],mass=particle.Particle.from_pdgid(p[lepton]).mass/1000)
            dR=[]
            for j in true_photon:
                temp=np.sqrt((eta[lepton]-eta[j])**2+(phi[lepton]-phi[j])**2)
                dR.append(temp)
                if temp<0.1:
                    y=vector.obj(pt=pt[j],eta=eta[j],phi=phi[j],mass=0)
                    dressed_lepton=dressed_lepton+y

                    #print(dressed_lepton.pt,dressed_lepton.eta,dressed_lepton.phi)
                    candidate_photons=np.delete(candidate_photons,np.where(candidate_photons==j)[0])
            dressed_leptons.append(dressed_lepton)
            for j in candidate_photons:
                temp=np.sqrt((eta[lepton]-eta[j])**2+(phi[lepton]-phi[j])**2)
                if temp<0.4:
                    candidate_photons=np.delete(candidate_photons,np.where(candidate_photons==j)[0])

        #print(dressed_leptons,len(candidate_photons),max(dressed_leptons,key=lambda v: v.pt).pt)

        if len(candidate_photons)>0:
            #print('here')
            #photon_pt=pt[candidate_photons]
            events_df.at[i,'Nu_y_coord']=[true_neutrino_lepton[np.argmax(neutrino_pt)],candidate_photons[np.argmax(pt[candidate_photons])]]
            events_df.at[i,'Charged_lepton']=[p[true_ch_lepton[np.argmax(ch_lepton_pt)]]]
            events_df.at[i,'Dressed_lepton.PT']=max(dressed_leptons,key=lambda v: v.pt).pt
            events_df.at[i,'Dressed_lepton.Eta']=max(dressed_leptons,key=lambda v: v.pt).eta
            events_df.at[i,'Dressed_lepton.Phi']=max(dressed_leptons,key=lambda v: v.pt).phi
            #print(events_df.iloc[i])

        else:
            events_df.at[i,'Nu_y_coord']=[true_neutrino_lepton[np.argmax(neutrino_pt)],None]
            events_df.at[i,'Charged_lepton']=[p[true_ch_lepton[np.argmax(ch_lepton_pt)]]]
            events_df.at[i,'Dressed_lepton.PT']=max(dressed_leptons,key=lambda v: v.pt).pt
            events_df.at[i,'Dressed_lepton.Eta']=max(dressed_leptons,key=lambda v: v.pt).eta
            events_df.at[i,'Dressed_lepton.Phi']=max(dressed_leptons,key=lambda v: v.pt).phi










        """
        if len(true_photon)>0:
                photon_pt=pt[true_photon]
                events_df.at[i,'Particle_coord']=[true_ch_lepton[np.argmax(ch_lepton_pt)],true_neutrino_lepton[np.argmax(neutrino_pt)],true_photon[np.argmax(photon_pt)]]
                events_df.at[i,'Charged_lepton']=[p[true_ch_lepton[np.argmax(ch_lepton_pt)]]]
                #print('ch lepton',true_ch_lepton[np.argmax(ch_lepton_pt)],np.max(ch_lepton_pt), 'neutrino',true_neutrino_lepton[np.argmax(neutrino_pt)],np.max(neutrino_pt),'photon',true_photon[np.argmax(photon_pt)],np.max(photon_pt)  )

        else:
            events_df.at[i,'Particle_coord']=[true_ch_lepton[np.argmax(ch_lepton_pt)],true_neutrino_lepton[np.argmax(neutrino_pt)],None ]
            events_df.at[i,'Charged_lepton']=[p[true_ch_lepton[np.argmax(ch_lepton_pt)]]]
            #print('ch lepton',true_ch_lepton[np.argmax(ch_lepton_pt)],np.max(ch_lepton_pt), 'neutrino',true_neutrino_lepton[np.argmax(neutrino_pt)],np.max(neutrino_pt),'no photon' )


        #print('ch lepton index = ',true_ch_lepton[np.argmax(ch_lepton_pt)],'neutrino index = ',true_neutrino_lepton[np.argmax(neutrino_pt)],'photon index = ',true_photon[np.argmax(photon_pt)])
        #print(np.intersect1d(photon,status_mask))"""
        
    
    

    particle_level_df=pd.DataFrame(None, index=range(len(events_df)),columns=['Muon.PT','Muon.Phi','Muon.Eta','Muon.Charge',
                                                                            'Electron.PT','Electron.Phi','Electron.Eta','Electron.Charge',
                                                                            'MissingET.MET_recon','MissingET.Phi_recon','MissingET.Eta_recon',
                                                                            'MissingET.MET','MissingET.Phi','MissingET.Eta',
                                                                            'Muon_size','Electron_size','MissingET_size',
                                                                            'Photon.PT','Photon.Phi','Photon.Eta','Photon_size',
                                                                            'Jet_size','Jet.PT','Jet.Eta','Jet.Phi'])

    for i in range(len(events_df)):
    
        p=events_df['Particle.PID'][i]
        pt=events_df['Particle.PT'][i]
        eta=events_df['Particle.Eta'][i]
        phi=events_df['Particle.Phi'][i]



        particle_level_df.loc[i,'MissingET.MET']=events_df['GenMissingET.MET'][i][0]
        particle_level_df.loc[i,'MissingET.Eta']=events_df['GenMissingET.Eta'][i][0]
        particle_level_df.loc[i,'MissingET.Phi']=events_df['GenMissingET.Phi'][i][0]




        particle_level_df.loc[i,'Jet_size']=events_df['GenJet_size'][i]
        particle_level_df.at[i,'Jet.PT']=events_df['GenJet.PT'][i]
        particle_level_df.at[i,'Jet.Eta']=events_df['GenJet.Eta'][i]
        particle_level_df.at[i,'Jet.Phi']=events_df['GenJet.Phi'][i]

        nu_y_coord=events_df['Nu_y_coord'][i]
        
        lepton=events_df['Charged_lepton'][i]
        if np.abs(lepton[0])==11:
            particle_level_df.loc[i,'Electron.PT']=events_df['Dressed_lepton.PT'][i]
            particle_level_df.loc[i,'Electron.Eta']=events_df['Dressed_lepton.Eta'][i]
            particle_level_df.loc[i,'Electron.Phi']=events_df['Dressed_lepton.Phi'][i]
            particle_level_df.loc[i,'Electron.Charge']=-int(np.sign(lepton[0]))
            particle_level_df.loc[i,'Electron_size']=1
            particle_level_df.loc[i,'Muon_size']=0
        elif np.abs(lepton[0])==13:
            particle_level_df.loc[i,'Muon.PT']=events_df['Dressed_lepton.PT'][i]
            particle_level_df.loc[i,'Muon.Eta']=events_df['Dressed_lepton.Eta'][i]
            particle_level_df.loc[i,'Muon.Phi']=events_df['Dressed_lepton.Phi'][i]
            particle_level_df.loc[i,'Muon.Charge']=-int(np.sign(lepton[0]))
            particle_level_df.loc[i,'Electron_size']=0
            particle_level_df.loc[i,'Muon_size']=1

        neutrino=nu_y_coord[0]
        photon=nu_y_coord[1]
        if photon==None:
            continue
        else:
            particle_level_df.loc[i,'Photon.PT']=pt[photon]
            particle_level_df.loc[i,'Photon.Eta']=eta[photon]
            particle_level_df.loc[i,'Photon.Phi']=phi[photon]
            particle_level_df.loc[i,'Photon_size']=1


        particle_level_df.loc[i,'MissingET.MET_recon']=pt[neutrino]
        particle_level_df.loc[i,'MissingET.Eta_recon']=eta[neutrino]
        particle_level_df.loc[i,'MissingET.Phi_recon']=phi[neutrino]



        particle_level_df.loc[i,'MissingET_size']=1
        
        particle_level_df.loc[i,'Photon.PT']=pt[photon]
        particle_level_df.loc[i,'Photon.Eta']=eta[photon]
        particle_level_df.loc[i,'Photon.Phi']=phi[photon]
        particle_level_df.loc[i,'Photon_size']=1



        '''

        if nu_y_coord!= None:
            
            #lepton=particles[0]
            neutrino=nu_y_coord[0]
            photon_s=nu_y_coord[1]

            if photon_s!=None:

                particle_level_df.loc[i,'MissingET.MET_recon']=pt[neutrino]
                particle_level_df.loc[i,'MissingET.Eta_recon']=eta[neutrino]
                particle_level_df.loc[i,'MissingET.Phi_recon']=phi[neutrino]



                particle_level_df.loc[i,'MissingET_size']=1
                
                particle_level_df.loc[i,'Photon.PT']=pt[photon_s]
                particle_level_df.loc[i,'Photon.Eta']=eta[photon_s]
                particle_level_df.loc[i,'Photon.Phi']=phi[photon_s]
                particle_level_df.loc[i,'Photon_size']=1
                

                if np.abs(p[lepton])==11:
                    particle_level_df.loc[i,'Electron.PT']=pt[lepton]
                    particle_level_df.loc[i,'Electron.Eta']=eta[lepton]
                    particle_level_df.loc[i,'Electron.Phi']=phi[lepton]
                    particle_level_df.loc[i,'Electron.Charge']=-int(np.sign(p[lepton]))
                    particle_level_df.loc[i,'Electron_size']=1
                    particle_level_df.loc[i,'Muon_size']=0

                    

                elif np.abs(p[lepton])==13:
                    particle_level_df.loc[i,'Muon.PT']=pt[lepton]
                    particle_level_df.loc[i,'Muon.Eta']=eta[lepton]
                    particle_level_df.loc[i,'Muon.Phi']=phi[lepton]
                    particle_level_df.loc[i,'Muon.Charge']=-int(np.sign(p[lepton]))
                    particle_level_df.loc[i,'Muon_size']=1
                    particle_level_df.loc[i,'Electron_size']=0
                    
            else:
                continue
        else: continue'''
        
        


    CMS_cut_part_initial=particle_level_df.copy()
    CMS_cut_part_initial=CMS_cut_part_initial.loc[ ((CMS_cut_part_initial['Photon_size']==1) & (CMS_cut_part_initial['MissingET_size']==1) & ((CMS_cut_part_initial['Muon_size']==1)) & ( CMS_cut_part_initial['Electron_size']==0 ) ) |
                            ((CMS_cut_part_initial['Photon_size']==1) & (CMS_cut_part_initial['MissingET_size']==1) & ((CMS_cut_part_initial['Electron_size']==1))   &( CMS_cut_part_initial['Muon_size']==0) )  ].copy()


    CMS_cut_part_initial = CMS_cut_part_initial.map(extract_single_element)
    CMS_cut_part=CMS_cut_func(CMS_cut_part_initial)

    cross_section_inclusive=ufloat(0.9919,0.0016)
    FILE_SIZE=len(events_df)

    #138*995.7/np.sum(events_df_part_full['Event.Weight'])


    e=len(CMS_cut_part)/FILE_SIZE
    epsilon=ufloat(   e, np.sqrt( e*(1-e)/FILE_SIZE  )  )

    cross_section_integrated=cross_section_inclusive*epsilon *1000


    fig,axes=plt.subplots(2,2,figsize=(12,10))


    Pt_lepton=pd.concat([CMS_cut_part['Muon.PT'],CMS_cut_part['Electron.PT']],axis=0,ignore_index=True)
    var=Pt_lepton
    #weight=pd.concat([CMS_cut_part['Event.Weight'],CMS_cut_part['Event.Weight']],axis=0,ignore_index=True)

    counts, bins = np.histogram(var, 25,range=(30, 140))#,weights=weight)
    bin_centres = (bins[:-1] + bins[1:])/2.
    err = np.sqrt(counts)
    sns.histplot(x=var,bins=25,element='step',binrange=[30,140],edgecolor='black',ax=axes[0][0])#,weights=weight)
    axes[0][0].errorbar(bin_centres,counts,yerr=err,fmt='o',color='black',markersize=5,elinewidth=2.5,label='Data')

    axes[0][0].set_xlim(30,140)
    axes[0][0].set_ylabel('Counts / 4.4 GeV')
    axes[0][0].set_xlabel(r'$\mathrm{p_{T}^{l}}$'+' (GeV)')




    Pt_photon=CMS_cut_part['Photon.PT']
    var=Pt_photon

    counts, bins = np.histogram(var, 9,range=(30, 1000))#,weights=CMS_cut_part['Event.Weight'])
    bin_centres = (bins[:-1] + bins[1:])/2.
    err = np.sqrt(counts)
    sns.histplot(x=var,bins=9,element='step',edgecolor='black',binrange=[30,1000],ax=axes[0][1])#,weights=CMS_cut_part['Event.Weight'])
    axes[0][1].errorbar(bin_centres,counts,yerr=err,fmt='o',color='black',markersize=5,elinewidth=2.5,label='Data')
    axes[0][1].set_yscale('log')
    axes[0][1].set_ylabel('Counts / 12.2 GeV')
    axes[0][1].set_xlabel(r'$\mathrm{p_{T}^{\gamma}}$'+' (GeV)')
    axes[0][1].set_xlim(30,1000)






    var=CMS_cut_part['Mt_lv']

    counts, bins = np.histogram(var, 30,range=(0, 200))#,weights=CMS_cut_part['Event.Weight'])
    bin_centres = (bins[:-1] + bins[1:])/2.
    err = np.sqrt(counts)
    sns.histplot(x=var,bins=30,element='step',edgecolor='black',binrange=[0,200],ax=axes[1][0])#,weights=CMS_cut_part['Event.Weight'])
    axes[1][0].errorbar(bin_centres,counts,yerr=err,fmt='o',color='black',markersize=5,elinewidth=2.5,label='Data')
    axes[1][0].set_ylabel('Counts / 6.6 GeV')
    axes[1][0].set_xlabel(r'$\mathrm{m_{T}(l,p_{T}^{miss})}$'+' (GeV)')






    var=CMS_cut_part['Mt_cluster']

    counts, bins = np.histogram(var, 50)#,weights=CMS_cut_part['Event.Weight'])
    bin_centres = (bins[:-1] + bins[1:])/2.
    err = np.sqrt(counts)
    sns.histplot(x=var,bins=50,element='step',edgecolor='black',ax=axes[1][1])#,weights=CMS_cut_part['Event.Weight'])
    axes[1][1].errorbar(bin_centres,counts,yerr=err,fmt='o',color='black',markersize=5,elinewidth=2.5,label='Data')
    axes[1][1].set_yscale('log')
    axes[1][1].set_ylabel('Counts / 50 GeV')
    axes[1][1].set_xlabel(r'$\mathrm{m_{T}^{l \gamma \nu}}$'+' (GeV)')

    plt.suptitle('CMS Valdiation Particle level',size=25)


    plt.tight_layout()
    plt.savefig(part_output_plot,dpi=800)

    CMS_cut_part.to_csv(part_output_csv)




def main():
    input_file=sys.argv[1]
    

    det_output_csv=sys.argv[2]
    part_output_csv=sys.argv[3]
    det_output_plot=sys.argv[4]
    part_output_plot=sys.argv[5]
    


    analysis(input_file,det_output_csv,det_output_plot,part_output_plot,part_output_csv)
if __name__=="__main__":
    main()



                



