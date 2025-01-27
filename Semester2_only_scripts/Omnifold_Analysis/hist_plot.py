import numpy as np
import uncertainties
import uncertainties.unumpy as unp
def hist(data,weights,bins,constant,weights_full):
    
    ### Estimated yields
    hist,bin=np.histogram(data,weights=weights*constant,bins=bins) ## aka sum of weights in bin j
    bin_centres = (bin[:-1] + bin[1:]) / 2
    bin_widths = bin[1:] - bin[:-1]

    hist_sq,_=np.histogram(data,weights=(weights*constant)**2,bins=bins) ## aka sum of weights**2 in bin j

    counts=hist
    
    counts_err=counts * np.sqrt(  (hist_sq)/hist**2   )

    expected_yield=unp.uarray(counts,np.abs(counts_err))

    area=np.sum(bin_widths*hist)

    counts_norm=counts/area

    counts_norm_err=counts_norm*np.sqrt(  (counts_err/counts)**2 +(np.sum(counts_err**2)/np.sum(counts)**2)  )


    expected_yield_norm=unp.uarray(counts_norm,np.abs(counts_norm_err))

    return {'expected':expected_yield,'expected_norm':expected_yield_norm,'centres':bin_centres}

def hist_plot(data,weights,bins,constant,weights_full):
    return hist(data,weights,bins,constant,weights_full)

if __name__=="__main__":

    print('Import successful')