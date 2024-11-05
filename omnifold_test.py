for i in range(itnum):

    w1=np.concatenate((np.ones(len(data_det)),push_series.loc[D]))

    model_step1 = Model(inputs=inputs, outputs=outputs)
    X_train,X_test,y_train,y_test,w_train,w_test=train_test_split(input_det[unfold_vars],
                                                                  input_det['target'],
                                                                  w1)
    
    model_step1.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

    model_step1.fit(X_train,y_train,sample_weight=w_train,epochs=40,batch_size=1000,validation_data=(X_test,y_test,w_test),verbose=False)

    f=model_step1.predict(sim_det.loc[sim_det['event'].isin(common_events)][unfold_vars],batch_size=1000)
    temp=f/(1.-f)
    temp=np.squeeze(np.nan_to_num(temp)) 

    pull_series.loc[common_events]=temp*push_series.loc[common_events]
    pull_series.loc[det_eff_events]=1
    
    
    


    #weights_pull=np.array(sim_part.loc[sim_part['event'].isin(common_events),'w'])*temp


    w2=np.concatenate((pull_series[P],push_series[P]))

    model_step2 = Model(inputs=inputs, outputs=outputs)

    X_train,X_test,y_train,y_test,w_train,w_test=train_test_split(input_gen[unfold_vars],
                                                                  input_gen['target'],
                                                                  w2)
    
    model_step2.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

    model_step2.fit(X_train,y_train,sample_weight=w_train,epochs=40,batch_size=1000,validation_data=(X_test,y_test,w_test),verbose=False)


    
    f=model_step2.predict(sim_part[unfold_vars],batch_size=1000)
    temp=f/(1.-f)
    temp=np.squeeze(np.nan_to_num(temp)) 
    
    
    push_series.loc[P]=temp*push_series[P]
    push_series.loc[neutrino_eff_events]=1



    
    

