from cmath import nan
import numpy as np
import pandas as pd
import tqdm
from numba import jit
from typing import Union, Optional, List, Sequence, Iterable, Mapping, Literal, Tuple


_Peakcalling_Method = Optional[Literal["test","MACS2","Blockify"]]
_reference = Optional[Literal["hg38","mm10","yeast"]]
_PeaktestMethod = Optional[Literal["poisson","binomial"]]

@jit(nopython=True)
def _findinsertionslen2(Chrom, start, end, length = 3, startpoint = 0,totallength = 10000000):
    # function to calculate the number of hops in the spcific area of chromosomes
    
    count = 0
    initial = startpoint
    flag = 0
    
    for i in range(startpoint,totallength):
        if Chrom[i] >= start-length and Chrom[i] <= end :
            if flag == 0:
                initial = i
                flag = 1
            count += 1
        elif Chrom[i] > end and count!= 0:
            return count,initial
        
    return count,initial

def _findinsertionslen(Chrom, start, end, length = 3):
    
    # function to calculate the number of hops in the spcific area of chromosomes
    return len(Chrom[(Chrom >= max(start-length,0)) & (Chrom <= end )])

def _findinsertions(Chrom, start, end, length = 3):

    # function returns of hops in the spcific area of chromosomes
    return Chrom[(Chrom >= max(start-length,0)) & (Chrom <= end)]

def _compute_cumulative_poisson(exp_hops_region,bg_hops_region,total_exp_hops,total_bg_hops,pseudocounts):
    
    from scipy.stats import poisson
    
    # Calculating the probability under the hypothesis of possion distribution
    if total_bg_hops >= total_exp_hops:
        return(1-poisson.cdf((exp_hops_region+pseudocounts),bg_hops_region * (float(total_exp_hops)/float(total_bg_hops)) + pseudocounts))
    else:
        return(1-poisson.cdf(((exp_hops_region *(float(total_bg_hops)/float(total_exp_hops)) )+pseudocounts),bg_hops_region + pseudocounts))

def _testCompare_bf2(
    bound: list, 
    curChromnp: np.ndarray, 
    curframe: np.ndarray, 
    length: int, 
    lam_win_size: Optional[int],  
    boundnew: list, 
    pseudocounts: float = 0.2, 
    pvalue_cutoff: float  = 0.00001, 
    chrom: str = None, 
    test_method: _PeaktestMethod = "poisson", 
    record: bool = True) -> list:
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
        
    # test whether the potiential peaks are true peaks by comparing to other data
    
    startpointTTAA = 0
    
    if lam_win_size != None:
        startpointTTAAlam = 0
        startpointboundlam = 0
        
        
    totallengthcurframe = len(curframe)
    totallengthcurChromnp = len(curChromnp)
    
    for i in range(len(bound)):
        
        # calculate the total number of hops in total
        TTAAnum, startpointTTAA = _findinsertionslen2(curframe, bound[i][0], bound[i][1], length, startpointTTAA ,totallengthcurframe) 
        boundnum = bound[i][2]
        
        if lam_win_size == None:
            
            scaleFactor = float(totallengthcurChromnp/totallengthcurframe)
            lam = TTAAnum * scaleFactor +pseudocounts
            
            if test_method == "poisson":
                pvalue = 1-poisson.cdf(boundnum , lam)
            elif test_method == "binomial":
                pvalue = binomtest(int(boundnum+pseudocounts), n=totallengthcurChromnp, 
                                   p=((TTAAnum+pseudocounts)/totallengthcurframe) , alternative='greater').pvalue
                
        else:
            
            TTAAnumlam,startpointTTAAlam = _findinsertionslen2(curframe, bound[i][0] - lam_win_size/2 + 1, 
                                                             bound[i][1] + lam_win_size/2, length, 
                                                             startpointTTAAlam, totallengthcurframe) 
            boundnumlam,startpointboundlam = _findinsertionslen2(curChromnp, bound[i][0] - lam_win_size/2 + 1, 
                                                               bound[i][1] + lam_win_size/2, length,
                                                              startpointboundlam, totallengthcurChromnp) 
            
         
            scaleFactor = float(boundnumlam/TTAAnumlam)
            lam = TTAAnum * scaleFactor +pseudocounts
            
            if test_method == "poisson":  
                pvalue = 1-poisson.cdf(boundnum , lam)
            elif test_method == "binomial":
                pvalue = binomtest(int(boundnum+pseudocounts), n=boundnumlam, 
                                   p=((TTAAnum+pseudocounts)/TTAAnumlam) , alternative='greater').pvalue
        
        if pvalue < pvalue_cutoff:
            if record:
                boundnew.append([chrom, bound[i][0], bound[i][1], boundnum, TTAAnum, lam, pvalue])
            else:
                boundnew.append([chrom, bound[i][0], bound[i][1]])
                
    return boundnew

def _testCompare2(
    bound: list, 
    curChromnp : np.ndarray,
    curbgframe: np.ndarray,
    curTTAAframenp: np.ndarray,
    length: int,
    lam_win_size: Optional[int] ,  
    boundnew: list,
    pseudocounts: float, 
    pvalue_cutoffbg: float, 
    pvalue_cutoffTTAA: float, 
    chrom: str, 
    test_method: _PeaktestMethod, 
    record: bool) -> list:
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
    # test whether the potiential peaks are true peaks by comparing to other data
    
    startbg = 0
    startTTAA = 0
    
    totalcurChrom = len(curChromnp)
    totalcurbackground = len(curbgframe)
    totalcurTTAA = len(curTTAAframenp)
    
    if lam_win_size != None:
        
        startbglam = 0
        startTTAAlam = 0
        startboundlam = 0
        
    for i in range(len(bound)):
        
        
        bgnum, startbg = _findinsertionslen2(curbgframe, bound[i][0], bound[i][1], length, startbg, totalcurbackground ) 
        TTAAnum, startTTAA = _findinsertionslen2(curTTAAframenp,bound[i][0], bound[i][1], length, startTTAA, totalcurTTAA) 
        boundnum = bound[i][2]

        
        if lam_win_size == None:

            
            scaleFactorTTAA = totalcurChrom/totalcurTTAA
            lamTTAA = TTAAnum * scaleFactorTTAA +pseudocounts
            
            scaleFactorbg = totalcurChrom/totalcurbackground
            lambg = TTAAnum * scaleFactorbg +pseudocounts
            
            if test_method == "poisson":
                
                pvalueTTAA = 1-poisson.cdf(boundnum , lamTTAA)
                pvaluebg = _compute_cumulative_poisson(boundnum,bgnum,totalcurChrom,totalcurbackground,pseudocounts)
                
            elif test_method == "binomial":
                
                pvalueTTAA = binomtest(int(boundnum+pseudocounts), n=totalcurChrom, 
                                   p=((TTAAnum+pseudocounts)/totalcurTTAA ) , alternative='greater').pvalue
                pvaluebg = binomtest(int(boundnum+pseudocounts), n=totalcurChrom, 
                                   p=((bgnum+pseudocounts)/totalcurbackground) , alternative='greater').pvalue
            
        else:


            bgnumlam, startbglam = _findinsertionslen2(curbgframe, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2, 
                                                      length, startbglam, totalcurbackground)
            TTAAnumlam, startTTAAlam = _findinsertionslen2(curTTAAframenp, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2,
                                                       length, startTTAAlam, totalcurTTAA) 
            boundnumlam, startboundlam = _findinsertionslen2(curChromnp, bound[i][0] - lam_win_size/2 + 1, bound[i][1] + lam_win_size/2, 
                                             length, startboundlam, totalcurChrom) 
            
            scaleFactorTTAA = boundnumlam/TTAAnumlam
            lamTTAA = TTAAnum * scaleFactorTTAA +pseudocounts
            
            
            if bgnumlam!= 0:
                scaleFactorbg = boundnumlam/bgnumlam
                lambg = bgnum * scaleFactorbg +pseudocounts
            else:
                lambg = 0
            
            if test_method == "poisson":
                
                
                pvalueTTAA = 1-poisson.cdf(boundnum , lamTTAA)
                pvaluebg = _compute_cumulative_poisson(boundnum,bgnum,boundnumlam,bgnumlam,pseudocounts)
                
            elif test_method == "binomial":
                

                pvalueTTAA = binomtest(int(boundnum+pseudocounts), n=boundnumlam, 
                                   p=((TTAAnum+pseudocounts)/TTAAnumlam ) , alternative='greater').pvalue


                if bgnumlam == 0:
                    pvaluebg = 0
                else:
                    pvaluebg = binomtest(int(boundnum+pseudocounts), n=boundnumlam, 
                                   p=((bgnum+pseudocounts)/bgnumlam) , alternative='greater').pvalue
   
        
        if pvaluebg < pvalue_cutoffbg and pvalueTTAA < pvalue_cutoffTTAA :
            
            if record:
                boundnew.append([chrom, bound[i][0], bound[i][1], boundnum, bgnum, TTAAnum, lambg, lamTTAA, pvaluebg, pvalueTTAA])
            else:
                boundnew.append([chrom, bound[i][0], bound[i][1]])
                
    return boundnew

def _test_bf2(
    expdata: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    pvalue_cutoff: float = 0.01,  
    mininser: int = 5, 
    minlen: int =  0,
    extend: int = 150, 
    maxbetween: int = 2800,  
    lam_win_size: Optional[int] =None,  
    pseudocounts: float = 0.2, 
    test_method: _PeaktestMethod = "poisson", 
    record: bool = False) -> pd.DataFrame:
    

    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):

        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        if len(curTTAAframe) == 0:
            continue
            
        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        
        

        # make a summary of our current insertion start points
        unique, counts = np.unique(curChromnp, return_counts=True)

        # create a list to find out the protintial peak region of their bounds
        bound = []

        # initial the start point, end point and the totol number of insertions
        startbound = 0
        endbound = 0
        insertionbound = 0

        # calculate the distance between each points
        dif1 = np.diff(unique, axis=0)
        # add a zero to help end the following loop at the end
        dif1 = np.concatenate((dif1,np.array([maxbetween+1])))


        # look for the uique insertion points
        for i in range(len(unique)):
            if startbound == 0:
                startbound = unique[i]
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i] 
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0 
            else:
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i]
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0

        
        boundnew = _testCompare_bf2(bound, curChromnp, curTTAAframe, length, lam_win_size,  boundnew,  pseudocounts, 
                                  pvalue_cutoff, chrom,  test_method = test_method,record = record)

        
    if record:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End", "Experiment Hops", "Reference Hops", "Expected Hops", "pvalue"])

    else:
        #print(boundnew)
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End"])
    
def _test2(
    expdata: pd.DataFrame, 
    backgroundframe: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    pvalue_cutoffbg: float = 0.00001, 
    pvalue_cutoffTTAA: float= 0.000001,  
    mininser: int = 5, 
    minlen: int = 0,
    extend: int = 150, 
    maxbetween: int = 2800,  
    lam_win_size: Optional[int] =None,  
    pseudocounts: float = 0.2, 
    test_method:  _PeaktestMethod = "poisson", 
    record: bool = False)-> pd.DataFrame:

    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):
        
        curbackgroundframe = np.array(list(backgroundframe[backgroundframe[0]==chrom][1]))
        if len(curbackgroundframe) == 0:
            continue
        curbackgroundframe.sort()
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        if len(curTTAAframe) == 0:
            continue

        
        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        

            

        # make a summary of our current insertion start points
        unique, counts = np.unique(np.array(curChrom), return_counts=True)

        # create a list to find out the protintial peak region of their bounds
        bound = []

        # initial the start point, end point and the totol number of insertions
        startbound = 0
        endbound = 0
        insertionbound = 0

        # calculate the distance between each points
        dif1 = np.diff(unique, axis=0)
        # add a zero to help end the following loop at the end
        dif1 = np.concatenate((dif1,np.array([maxbetween+1])))


        # look for the uique insertion points
        for i in range(len(unique)):
            if startbound == 0:
                startbound = unique[i]
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i] 
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0 
            else:
                insertionbound += counts[i]
                if dif1[i] > maxbetween :
                    endbound = unique[i]
                    if (insertionbound >= mininser) and ((endbound- startbound) >= minlen):
                        bound.append([max(startbound-extend,0),endbound+4+extend,insertionbound, endbound-startbound])
                    startbound = 0
                    endbound = 0
                    insertionbound = 0

                               
        boundnew = _testCompare2(bound, curChromnp, curbackgroundframe, curTTAAframe, length, lam_win_size, boundnew, pseudocounts, pvalue_cutoffbg, pvalue_cutoffTTAA, chrom, test_method , record)
 
        
    if record:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End", "Experiment Hops", 
                                                   "Background Hops", "Reference Hops", "Expected Hops background", "Expected Hops Reference", 
                                                       "pvalue Background", "pvalue Reference"])

    else:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End"])

def _BlockifyCompare(
    bound: list, 
    curChrom: np.ndarray, 
    curframe: np.ndarray, 
    length: int, 
    boundnew: list, 
    scaleFactor: float, 
    pseudocounts: float, 
    pvalue_cutoff: float, 
    chrom: str, 
    test_method:  _PeaktestMethod = "poisson", 
    record: bool = True) -> list:
# test whether the potiential peaks are true peaks by comparing to TTAAs

    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
    last = -1
    Chrnumtotal = 0
    TTAAnumtotal = 0

    for i in range(len(bound)):
        TTAAnum = _findinsertionslen(curframe, bound[i][0], bound[i][1], length)
        boundnum = bound[i][2]
        
        if test_method == "poisson":
            pValue = 1-poisson.cdf(boundnum - 1, TTAAnum * scaleFactor+pseudocounts)
            
        elif test_method == "binomial":
            pValue = binomtest(int(boundnum+pseudocounts), n=len(curChrom), 
                               p=((TTAAnum+pseudocounts)/len(curframe) ) , alternative='greater').pvalue
        

        if pValue <= pvalue_cutoff and last == -1 :
            
            last = i
            Chrnumtotal += boundnum
            TTAAnumtotal += TTAAnum

        elif pValue > pvalue_cutoff and last != -1 :
            
            if record:
                
                if test_method == "poisson":
                    pvalue = 1-poisson.cdf(Chrnumtotal - 1, TTAAnumtotal * scaleFactor+pseudocounts)

                elif test_method == "binomial":
                    pvalue = binomtest(int(Chrnumtotal+pseudocounts), n=len(curChrom), 
                                       p=((TTAAnumtotal+pseudocounts)/len(curframe) ) , alternative='greater').pvalue
                    
                boundnew.append([chrom, bound[last][0], bound[i-1][1], Chrnumtotal, TTAAnumtotal, TTAAnumtotal*scaleFactor+pseudocounts, pvalue])
            else:
                boundnew.append([chrom, bound[last][0], bound[i-1][1]])
                
            last = -1
            Chrnumtotal = 0
            TTAAnumtotal = 0


    if last != -1:
        
        if record:
            
            if test_method == "poisson":
                pvalue = 1-poisson.cdf(Chrnumtotal - 1, TTAAnumtotal * scaleFactor+pseudocounts)

            elif test_method == "binomial":
                pvalue = binomtest(int(Chrnumtotal+pseudocounts), n=len(curChrom), 
                                   p=((TTAAnumtotal+pseudocounts)/len(curframe) ) , alternative='greater').pvalue

            boundnew.append([chrom, bound[last][0], bound[i-1][1], Chrnumtotal, TTAAnumtotal, TTAAnumtotal*scaleFactor+pseudocounts, pvalue])
        else:
            boundnew.append([chrom, bound[last][0], bound[i-1][1]])

    return boundnew

def _Blockify(
    expdata: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    pvalue_cutoff: float = 0.0001, 
    pseudocounts: float = 0.2, 
    test_method:  _PeaktestMethod = "poisson", 
    record: bool = True) -> pd.DataFrame:
    
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):
        

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))

        if len(curTTAAframe) == 0:
            continue
            
        # make a summary of our current insertion start points
        unique, counts = np.unique(curChromnp, return_counts=True)

        # create a list to find out the protintial peak region of their bounds
        bound = []


        import astropy.stats as astrostats
        hist, bin_edges = astrostats.histogram(expdata[expdata[0] == chrom][1], bins="blocks")

        hist = list(hist)
        bin_edges = list(bin_edges.astype(int))
        for bins in range(len(bin_edges)-1):
            bound.append([bin_edges[bins],bin_edges[bins+1], hist[bins], bin_edges[bins+1]-bin_edges[bins]])


        
        boundnew = _BlockifyCompare(bound, curChromnp, curTTAAframe, length, 
                                   boundnew, scaleFactor = len(curChromnp)/len(curTTAAframe), pseudocounts = pseudocounts, 
                                   pvalue_cutoff = pvalue_cutoff, chrom = chrom, test_method = test_method, record = record)

        
    if record:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End", "Experiment Hops", "Reference Hops", "Expected Hops", "pvalue"])

    else:
        return pd.DataFrame(boundnew, columns=["Chr","Start", "End"]) 
    
def _callpeaksMACS2(
    expdata: pd.DataFrame, 
    background: pd.DataFrame,  
    TTAAframe: pd.DataFrame, 
    length: int,
    window_size: int = 1000,
    lam_win_size: Optional[int] =100000,
    step_size: int = 500,
    pseudocounts: float = 0.2,
    pvalue_cutoff: float = 0.01,
    record: bool = False) -> pd.DataFrame:
    
    # function for MACS2 with background 
    from scipy.stats import poisson
    
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())
    
    chr_list = []
    start_list = []
    end_list = []
    list_of_l_names = ["bg","1k","5k","10k"]
    pvalue_list = []
    
    if record:
        center_list = []
        num_exp_hops_list = []
        num_bg_hops_list = []
        frac_exp_list = []
        tph_exp_list = []
        frac_bg_list = []
        tph_bg_list = []
        tph_bgs_list = []
        lambda_type_list =[]
        lambda_list = []
        lambda_hop_list = []
        
        
    total_experiment_hops = len(expdata)
    total_background_hops = len(background)
    

    for chrom in tqdm.tqdm(chrm):

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChrom.sort()
        
        max_pos = curChrom[-1] + 4
        sig_start = 0
        sig_end = 0
        sig_flag = 0
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        
        curbackgroundframe = np.array(list(background[background[0]==chrom][1]))
        
        sig_start = 0
        sig_end = 0
        sig_flag = 0
        
        for window_start in range(0,int(max_pos+window_size),int(step_size)):

            num_exp_hops = _findinsertionslen(curChromnp, window_start, window_start+window_size - 1, length)
            if num_exp_hops > 1:
                num_bg_hops = _findinsertionslen(curbackgroundframe, window_start, window_start+window_size - 1, length)
                p = _compute_cumulative_poisson(num_exp_hops,num_bg_hops,total_experiment_hops,total_background_hops,pseudocounts)
            else:
                p = 1
                

            #is this window significant?
            if p < pvalue_cutoff:
                #was last window significant?
                if sig_flag:
                    #if so, extend end of windows
                    sig_end = window_start+window_size-1
                else:
                    #otherwise, define new start and end and set flag
                    sig_start = window_start
                    sig_end = window_start+window_size-1
                    sig_flag = 1

            else:
                #current window not significant.  Was last window significant?
                if sig_flag:
                    
                    #add full sig window to the frame of peaks 
                    #add chr, peak start, peak end
                    chr_list.append(chrom) #add chr to frame
                    start_list.append(sig_start) #add peak start to frame
                    end_list.append(sig_end) #add peak end to frame
                    
                    #compute peak center and add to frame
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    peak_center = np.median(overlap)

                    #add number of experiment hops in peak to frame
                    num_exp_hops = len(overlap)

                    #add number of background hops in peak to frame
                    num_bg_hops = _findinsertionslen(curbackgroundframe, sig_start, sig_end, length)
                    
                  
                    if record:
                        center_list.append(peak_center) #add peak center to frame
                        num_exp_hops_list.append(num_exp_hops)
                        #add fraction of experiment hops in peak to frame
                        frac_exp_list.append(float(num_exp_hops)/total_experiment_hops)
                        tph_exp_list.append(float(num_exp_hops)*100000/total_experiment_hops)
                        num_bg_hops_list.append(num_bg_hops)
                        frac_bg_list.append(float(num_bg_hops)/total_background_hops)
                        tph_bg_list.append(float(num_bg_hops)*100000/total_background_hops)
                        

                    #find lambda and compute significance of peak
                    if total_background_hops >= total_experiment_hops: #scale bg hops down
                        #compute lambda bg
                        num_TTAAs = _findinsertionslen(curTTAAframe, sig_start, sig_end, length)
                        lambda_bg = ((num_bg_hops*(float(total_experiment_hops)/total_background_hops))/max(num_TTAAs,1)) 


                        #compute lambda 1k
                        num_bg_hops_1k = _findinsertionslen(curbackgroundframe, peak_center-499, peak_center+500, length)
                        num_TTAAs_1k = _findinsertionslen(curTTAAframe, peak_center-499, peak_center+500, length)
                        lambda_1k = (num_bg_hops_1k*(float(total_experiment_hops)/total_background_hops))/(max(num_TTAAs_1k,1))


                        #compute lambda 5k
                        num_bg_hops_5k = _findinsertionslen(curbackgroundframe, peak_center-2499, peak_center+2500, length)
                        num_TTAAs_5k = _findinsertionslen(curTTAAframe, peak_center-2499, peak_center+2500, length)
                        lambda_5k = (num_bg_hops_5k*(float(total_experiment_hops)/total_background_hops))/(max(num_TTAAs_5k,1))


                        #compute lambda 10k
                        num_bg_hops_10k = _findinsertionslen(curbackgroundframe, peak_center-4999, peak_center+5000, length)
                        num_TTAAs_10k = _findinsertionslen(curTTAAframe, peak_center-4999, peak_center+5000, length)
                        lambda_10k = (num_bg_hops_10k*(float(total_experiment_hops)/total_background_hops))/(max(num_TTAAs_10k,1))
                        lambda_f = max([lambda_bg,lambda_1k,lambda_5k,lambda_10k])


                        #record type of lambda used
                        index = [lambda_bg,lambda_1k,lambda_5k,lambda_10k].index(max([lambda_bg,lambda_1k,lambda_5k,lambda_10k]))
                        lambda_type_list.append(list_of_l_names[index])
                        #record lambda
                        lambda_list.append(lambda_f)
                        #compute pvalue and record it

                        pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),lambda_f*max(num_TTAAs,1)+pseudocounts)
                        pvalue_list.append(pvalue)


                        tph_bgs = float(num_exp_hops)*100000/total_experiment_hops-float(num_bg_hops)*100000/total_background_hops
                        
                        if record:
                            lambda_type_list.append(list_of_l_names[index])
                            lambda_list.append(lambda_f)
                            tph_bgs_list.append(tph_bgs)
                            lambda_hop_list.append(lambda_f*max(num_TTAAs,1))


                        index = [lambda_bg,lambda_1k,lambda_5k,lambda_10k].index(max([lambda_bg,lambda_1k,lambda_5k,lambda_10k]))
                        lambdatype = list_of_l_names[index]
                        #l = [pvalue,tph_bgs,lambda_f,lambdatype]

                    else: #scale experiment hops down
                        

                        #compute lambda bg
                        num_TTAAs = _findinsertionslen(curTTAAframe, sig_start, sig_end, length)
                        lambda_bg = (float(num_bg_hops)/max(num_TTAAs,1)) 


                        #compute lambda 1k
                        num_bg_hops_1k = _findinsertionslen(curbackgroundframe, peak_center-499, peak_center+500, length)
                        num_TTAAs_1k = _findinsertionslen(curTTAAframe, peak_center-499, peak_center+500, length)
                        lambda_1k = (float(num_bg_hops_1k)/(max(num_TTAAs_1k,1)))


                        #compute lambda 5k
                        num_bg_hops_5k = _findinsertionslen(curbackgroundframe, peak_center-2499, peak_center+2500, length)
                        num_TTAAs_5k = _findinsertionslen(curTTAAframe, peak_center-2499, peak_center+2500, length)
                        lambda_5k = (float(num_bg_hops_5k)/(max(num_TTAAs_5k,1)))


                        #compute lambda 10k
                        num_bg_hops_10k = _findinsertionslen(curbackgroundframe, peak_center-4999, peak_center+5000, length)
                        num_TTAAs_10k = _findinsertionslen(curTTAAframe, peak_center-4999, peak_center+5000, length)
                        lambda_10k = (float(num_bg_hops_10k)/(max(num_TTAAs_10k,1)))
                        lambda_f = max([lambda_bg,lambda_1k,lambda_5k,lambda_10k])


                        #record type of lambda used
                        index = [lambda_bg,lambda_1k,lambda_5k,lambda_10k].index(max([lambda_bg,
                                                                                      lambda_1k,lambda_5k,lambda_10k]))

                        #compute pvalue and record it
                        pvalue = 1-poisson.cdf(((float(total_background_hops)/total_experiment_hops)*num_exp_hops+ pseudocounts),lambda_f*max(num_TTAAs,1)+pseudocounts)
                        pvalue_list.append(pvalue)

                        tph_bgs = float(num_exp_hops)*100000/total_experiment_hops -float(num_bg_hops)*100000/total_background_hops

                        if record:
                            lambda_type_list.append(list_of_l_names[index])
                            lambda_list.append(lambda_f)
                            tph_bgs_list.append(tph_bgs)
                            lambda_hop_list.append(lambda_f*max(num_TTAAs,1))


                        index = [lambda_bg,lambda_1k,lambda_5k,lambda_10k].index(max([lambda_bg,
                                                                                      lambda_1k,lambda_5k,lambda_10k]))
                        lambdatype = list_of_l_names[index]
                        
                        


                    #number of hops that are a user-defined distance from peak center
                    sig_flag = 0
                        

    if record:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Center","Experiment Hops",
                "Fraction Experiment","TPH Experiment","Lambda Type",
                "Lambda","Poisson pvalue"])


        peaks_frame["Lambda Reference Hops"] = lambda_hop_list
        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Hops"] = num_exp_hops_list 
        peaks_frame["Fraction Experiment"] = frac_exp_list 
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["Reference Hops"] = num_bg_hops_list 
        peaks_frame["Fraction background"] = frac_bg_list
        peaks_frame["TPH background"] = tph_bg_list
        peaks_frame["TPH background subtracted"] = tph_bgs_list
        peaks_frame["Lambda Type"] = lambda_type_list
        peaks_frame["Lambda"] = lambda_list
        
        
    else:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["Poisson pvalue"] = pvalue_list
        

    #peaks_frame = peaks_frame[peaks_frame["Poisson pvalue"] <= pvalue_cutoff]
    
    
    if record:
        return peaks_frame
    else:                    
        return peaks_frame[['Chr','Start','End']]

def _callpeaksMACS2_bfnew2(
    expdata: pd.DataFrame, 
    TTAAframe: pd.DataFrame, 
    length: int, 
    min_hops: int = 3, 
    extend: int = 200,
    window_size: int = 1000, 
    step_size: int = 500, 
    pseudocounts: float = 0.2, 
    pvalue_cutoff: float = 0.01,
    lam_win_size: Optional[int] = None,
    record: bool = False, 
    test_method: _PeaktestMethod = "poisson") -> pd.DataFrame:
    
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
        
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())
    
    # create lists to record 
    chr_list = []
    start_list = []
    end_list = []
    pvalue_list = []
    sig_end = 0
    
    if record:
        center_list = []
        num_exp_hops_list = []
        frac_exp_list = []
        tph_exp_list = []
        background_hops = []
        expect_hops = []
        
    total_experiment_hops = len(expdata)
    
    
    for chrom in tqdm.tqdm(chrm):
        
        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        totalcurTTAA = len(curTTAAframe)
        if len(curTTAAframe) == 0:
            continue

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        
        max_pos = curChrom[-1]
        sig_start = 0
        sig_end = 0
        sig_flag = 0

        
        totalcurChrom = len(curChromnp)
        
        
        starthops1 = 0
        startTTAA1 = 0
      

        startTTAA2 = 0
        
        if lam_win_size != None:
            
            starthopslam1 = 0
            startTTAAlam1 = 0

            starthopslam2 = 0
            startTTAAlam2 = 0
            
        if totalcurTTAA != 0:

            # caluclate the ratio for TTAA and background 
            if lam_win_size == None:
                lambdacur = (totalcurChrom/totalcurTTAA) #expected ratio of hops per TTAA

            for window_start in range(curChrom[0],int(max_pos+2*window_size),step_size):

                if sig_end >= window_start:
                    continue 

                num_exp_hops, starthops1 = _findinsertionslen2(curChromnp, window_start, window_start+window_size - 1, 
                                                              length, starthops1, totalcurChrom)
                
                if num_exp_hops >= min_hops:
                    
                    num_TTAAs_window, startTTAA1 = _findinsertionslen2(curTTAAframe, window_start, window_start+window_size - 1, 
                                                                     length, startTTAA1, totalcurTTAA)
                    
                    #is this window significant?
                    if test_method == "poisson":
                        
                        if lam_win_size == None:
                            pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),lambdacur*max(num_TTAAs_window,1)+pseudocounts)
                        else:
                            num_TTAA_hops_lambda, startTTAAlam1 = _findinsertionslen2(curTTAAframe , 
                                                                                     window_start - int(lam_win_size/2) +1, 
                                                                                     window_start+window_size + int(lam_win_size/2) - 1, 
                                                                                     length, startTTAAlam1, totalcurTTAA)
                           
                            num_exp_hops_lambda, starthopslam1 = _findinsertionslen2(curChromnp , 
                                                                                    window_start - int(lam_win_size/2) +1, 
                                                                                     window_start+window_size + int(lam_win_size/2) - 1, 
                                                                                    length, starthopslam1, totalcurChrom)
                    
                            
                            pvalue = 1-poisson.cdf((num_exp_hops+pseudocounts),
                                         float(num_exp_hops_lambda/num_TTAA_hops_lambda)*max(num_TTAAs_window,1)+pseudocounts)
                            
                    elif test_method == "binomial":
                        
                        if lam_win_size == None:
                            pvalue = binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                           p=((num_TTAAs_window+pseudocounts)/totalcurTTAA) , alternative='greater').pvalue
                        else:
                            num_TTAA_hops_lambda, startTTAAlam1 = _findinsertionslen2(curTTAAframe , 
                                                                                     window_start - int(lam_win_size/2) +1, 
                                                                                     window_start+window_size + int(lam_win_size/2) - 1, 
                                                                                     length, startTTAAlam1, totalcurTTAA)
                            num_exp_hops_lambda, starthopslam1 = _findinsertionslen2(curChromnp , 
                                                                                    window_start - int(lam_win_size/2) +1, 
                                                                                     window_start+window_size + int(lam_win_size/2) - 1, 
                                                                                    length, starthopslam1, totalcurChrom)
                            pvalue = binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lambda, 
                                           p=((num_TTAAs_window+pseudocounts)/num_TTAA_hops_lambda) , alternative='greater').pvalue

                else:
                    pvalue = 1

                if pvalue < pvalue_cutoff :
                    
                    #was last window significant?
                    if sig_flag:
                        
                        #if so, extend end of windows
                        sig_end = window_start+window_size-1
                        
                    else:
                        
                        #otherwise, define new start and end and set flag
                        sig_start = window_start
                        sig_end = window_start+window_size-1
                        sig_flag = 1

                else:
                    
                    #current window not significant.  Was last window significant?
                    if sig_flag:

                        #compute peak center and add to frame
                        
                        overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                        peak_center = np.median(overlap)
                        
                        # redefine the overlap
                        sig_start = overlap.min() - extend
                        sig_end = overlap.max() + length + extend
                        overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                        num_exp_hops = len(overlap)
                

                        num_TTAAs_window, startTTAA2 = _findinsertionslen2(curTTAAframe, sig_start, sig_end, length, 
                                                                          startTTAA2, totalcurTTAA)
                      
                        
                        #num_TTAAs_window= _findinsertionslen(curTTAAframe, sig_start, sig_end, length)
                        #                                                  startTTAA2, totalcurTTAA)

                        #compute pvalue and record it  
                        if test_method == "poisson":
                            
                            if lam_win_size == None:
                                pvalue_list.append(1-poisson.cdf((num_exp_hops+pseudocounts),lambdacur*max(num_TTAAs_window,1)+pseudocounts))
                            else:
                                num_exp_hops_lam_win_size, starthopslam2  = _findinsertionslen2(curChromnp, peak_center-(lam_win_size/2-1), 
                                                                              peak_center+(lam_win_size/2), length, 
                                                                                starthopslam2 ,totalcurChrom)
                         
                                
                                num_TTAAs_lam_win_size , startTTAAlam2= _findinsertionslen2(curTTAAframe, peak_center-(lam_win_size/2-1), 
                                                                           peak_center+(lam_win_size/2), length, startTTAAlam2,
                                                                            totalcurTTAA)
                                
                           
                                lambda_lam_win_size = float(num_exp_hops_lam_win_size)/(max(num_TTAAs_lam_win_size,1))
                                pvalue_list.append(1-poisson.cdf((num_exp_hops+pseudocounts), lambda_lam_win_size*max(num_TTAAs_window,1)+pseudocounts))
                            
                        elif test_method == "binomial":
                            
                            if lam_win_size == None:
                                pvalue_list.append(binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                               p=((num_TTAAs_window+pseudocounts)/totalcurTTAA) , alternative='greater').pvalue)
                            else:
                                num_exp_hops_lam_win_size, starthopslam2  = _findinsertionslen2(curChromnp, peak_center-(lam_win_size/2-1), 
                                                                              peak_center+(lam_win_size/2), length, 
                                                                                starthopslam2 ,totalcurChrom)
                            
                                
                            
                                num_TTAAs_lam_win_size , startTTAAlam2= _findinsertionslen2(curTTAAframe, peak_center-(lam_win_size/2-1), 
                                                                           peak_center+(lam_win_size/2), length, startTTAAlam2,
                                                                            totalcurTTAA)
                                pvalue_list.append(binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lam_win_size, 
                                               p=((num_TTAAs_window+pseudocounts)/num_TTAAs_lam_win_size) , alternative='greater').pvalue)

                        chr_list.append(chrom) #add chr to frame
                        start_list.append(sig_start) #add peak start to frame
                        end_list.append(sig_end) #add peak end to frame


                        if record:

                            center_list.append(peak_center) #add peak center to frame
                            num_exp_hops_list.append(num_exp_hops)

                            #add fraction of experiment hops in peak to frame
                            frac_exp_list.append(float(num_exp_hops)/total_experiment_hops)
                            tph_exp_list.append(float(num_exp_hops)*100000/total_experiment_hops)
                      
                            background_hops.append(num_TTAAs_window)
                        
                            if lam_win_size == None:
                                expect_hops.append(lambdacur*max(num_TTAAs_window,1)+pseudocounts)
                            else:
                                expect_hops.append(float(num_exp_hops_lam_win_size)/(max(num_TTAAs_lam_win_size,1))*max(num_TTAAs_window,1)+pseudocounts)

                        sig_flag = 0


        if record:
            peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Center","pvalue","Experiment Hops","Reference Hops",
                    "Fraction Experiment","TPH Experiment"])

            peaks_frame["Center"] = center_list
            peaks_frame["Experiment Hops"] = num_exp_hops_list 
            peaks_frame["Fraction Experiment"] = frac_exp_list 
            peaks_frame["TPH Experiment"] = tph_exp_list
            peaks_frame["Reference Hops"] = background_hops
            peaks_frame["Expect Hops"] = expect_hops

        else:
            peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","pvalue"])

        
        peaks_frame["Chr"] = chr_list
        peaks_frame["Start"] = start_list
        peaks_frame["End"] = end_list
        peaks_frame["pvalue"] = pvalue_list
        peaks_frame = peaks_frame[peaks_frame["pvalue"] <= pvalue_cutoff]

    if record:
        return peaks_frame
    else:                    
        return peaks_frame[['Chr','Start','End']]

def _callpeaksMACS2new2(
    expdata: pd.DataFrame, 
    background: pd.DataFrame,  
    TTAAframe: pd.DataFrame, 
    length: int, 
    extend: int = 200, 
    lam_win_size: Optional[int] = 100000,
    pvalue_cutoff_background: float = 0.01,  
    pvalue_cutoff_TTAA: float = 0.01,
    window_size: int = 1000,
    step_size: int = 500,
    pseudocounts: float = 0.2,
    test_method: _PeaktestMethod = "poisson",
    min_hops: int = 3,
    record: bool = False) -> pd.DataFrame:
    
    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":  
        from scipy.stats import binomtest
    
    # The chromosomes we need to consider
    chrm = list(expdata[0].unique())
    
    # create lists to record the basic information
    chr_list = []
    start_list = []
    end_list = []
    pvalue_list_background = []
    pvalue_list_TTAA = []
    sig_end = 0
    
    if record:
        # create lists to record other information
        center_list = []
        num_exp_hops_list = []
        num_bg_hops_list = []
        num_TTAA_hops_list = []
        frac_exp_list = []
        tph_exp_list = []
        frac_bg_list = []
        tph_bg_list = []
        tph_bgs_list = []
        
    # record total number of hops  
    total_experiment_hops = len(expdata)
    total_background_hops = len(background)
    
    # going from the first Chromosome to the last
    for chrom in tqdm.tqdm(chrm):

        curbackgroundframe = np.array(list(background[background[0]==chrom][1]))
        totalcurbackground = len(curbackgroundframe)
        if totalcurbackground == 0:
            continue

        curTTAAframe = np.array(list(TTAAframe[TTAAframe[0]==chrom][1]))
        totalcurTTAA = len(curTTAAframe)
        if totalcurTTAA == 0:
            continue
        
        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata[0] == chrom][1]) 
        curChromnp = np.array(curChrom)
        
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        curbackgroundframe.sort()
        
        # initial the parameters
        max_pos = curChrom[-1] + length +1
        sig_start = 0
        sig_end = 0
        sig_flag = 0
        
        
        
        # calculate the total number of hops
        totalcurChrom = len(curChromnp)
        
        starthop1 = 0
        startTTAA1 = 0
        startbg1 = 0
        
        starthop2 = 0
        startTTAA2 = 0
        startbg2 = 0
   

        if lam_win_size == None:
        
        # caluclate the ratio for TTAA and background 
            lambdacurTTAA = float(totalcurChrom/totalcurTTAA) #expected ratio of hops per TTAA
            lambdacurbackground = float(totalcurChrom/totalcurbackground) #expected ratio of hops per background
            
        else:
            
            startTTAAlam1 = 0
            startbglam1 = 0
            starthoplam = 0
            
            startTTAAlam2 = 0
            startbglam2 = 0
            starthoplam2 = 0

        for window_start in range(curChromnp[0],int(max_pos+2*window_size),int(step_size)):

            if sig_end >= window_start:
                continue 

            num_exp_hops, starthop1 = _findinsertionslen2(curChromnp, window_start, window_start+window_size - 1, 
                                                        length, starthop1, totalcurChrom)
            

            if num_exp_hops >= min_hops :

                # find out the number of hops in the current window for backgound 
                num_bg_hops, startbg1 = _findinsertionslen2(curbackgroundframe, window_start, 
                                                           window_start+window_size - 1,length,
                                                           startbg1, totalcurbackground)


                if num_bg_hops >0 :

                    if lam_win_size == None:

                        if test_method == "poisson":
                            pvaluebg = _compute_cumulative_poisson(num_exp_hops,
                                                                  num_bg_hops,totalcurChrom,
                                                                  totalcurbackground,pseudocounts)
                        elif test_method == "binomial":
                            pvaluebg = binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                               p=((num_bg_hops+pseudocounts)/totalcurbackground) , 
                                                 alternative='greater').pvalue
                    else:

                        num_exp_hops_lam, starthoplam =_findinsertionslen2(curChromnp, 
                                                                window_start - int(lam_win_size/2) +1,
                                                                    window_start+window_size + int(lam_win_size/2) - 1, 
                                                                           length,starthoplam, totalcurChrom)
                        
                        num_exp_bg_lam, startbglam1 = _findinsertionslen2(curbackgroundframe, 
                                                             window_start - int(lam_win_size/2) +1,
                                                             window_start+window_size + int(lam_win_size/2) - 1,
                                                             length,startbglam1, totalcurbackground)

                        if test_method == "poisson":
                            pvaluebg = _compute_cumulative_poisson(num_exp_hops,
                                                                  num_bg_hops,num_exp_hops_lam,
                                                                  num_exp_bg_lam,pseudocounts)
                        elif test_method == "binomial":
                            pvaluebg = binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lam, 
                                               p=((num_bg_hops+pseudocounts)/num_exp_bg_lam) , 
                                                 alternative='greater').pvalue


                else:
                    if lam_win_size != None:
                        num_exp_hops_lam, starthoplam =_findinsertionslen2(curChromnp, 
                                                                window_start - int(lam_win_size/2) +1,
                                                                    window_start+window_size + int(lam_win_size/2) - 1, 
                                                                           length,starthoplam, totalcurChrom)
                        
                        num_exp_bg_lam, startbglam1 = _findinsertionslen2(curbackgroundframe, 
                                                             window_start - int(lam_win_size/2) +1,
                                                             window_start+window_size + int(lam_win_size/2) - 1,
                                                             length,startbglam1, totalcurbackground)

                    pvaluebg = 0

                # if it passes, then look at the TTAA:
                if pvaluebg < pvalue_cutoff_background :

                    num_TTAA_hops, startTTAA1 = _findinsertionslen2(curTTAAframe, window_start, 
                                                      window_start+window_size - 1, length,
                                                      startTTAA1, totalcurTTAA)

                    if lam_win_size == None:

                        if test_method == "poisson":
                            pvalueTTAA = 1-poisson.cdf((num_exp_hops+pseudocounts),
                                                       lambdacurTTAA*num_TTAA_hops+pseudocounts)
                        elif test_method == "binomial":
                            pvalueTTAA = binomtest(int(num_exp_hops+pseudocounts), n=totalcurChrom, 
                                               p=((num_TTAA_hops+pseudocounts)/totalcurTTAA) , 
                                                   alternative='greater').pvalue
                    else:


                        num_TTAA_hops_lam, startTTAAlam1 = _findinsertionslen2(curTTAAframe, 
                                                             window_start - int(lam_win_size/2) +1,
                                                             window_start+window_size + int(lam_win_size/2) - 1,
                                                             length, startTTAAlam1, totalcurTTAA)
                        if test_method == "poisson":
                            pvalueTTAA = 1-poisson.cdf((num_exp_hops+pseudocounts),
                                                       (num_exp_hops_lam/num_TTAA_hops_lam)*num_TTAA_hops+
                                                       pseudocounts)
                        elif test_method == "binomial":
                            pvalueTTAA = binomtest(int(num_exp_hops+pseudocounts), n=num_exp_hops_lam, 
                                               p=((num_TTAA_hops+pseudocounts)/num_TTAA_hops_lam) , 
                                                   alternative='greater').pvalue

                else:
                    pvaluebg = 1
                    pvalueTTAA = 1


            else:
                pvaluebg = 1
                pvalueTTAA = 1


            #is this window significant?
            if pvaluebg < pvalue_cutoff_background and pvalueTTAA < pvalue_cutoff_TTAA :
                #was last window significant?
                if sig_flag:
                    #if so, extend end of windows
                    sig_end = window_start+window_size-1
                else:
                    #otherwise, define new start and end and set flag
                    sig_start = window_start
                    sig_end = window_start+window_size-1
                    sig_flag = 1

            else:
                #current window not significant.  Was last window significant?
                if sig_flag:

                    # Let's first give a initial view of our peak
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    peak_center = np.median(overlap)

                    # redefine the overlap
                    sig_start = overlap.min() - extend
                    sig_end = overlap.max() + 3 + extend

                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    num_exp_hops = len(overlap)

                    #add number of background hops in peak to frame
                    num_TTAA_hops, startTTAA2 = _findinsertionslen2(curTTAAframe, sig_start, sig_end, 
                                                                      length, startTTAA2, totalcurTTAA)
                    num_bg_hops, startbg2 = _findinsertionslen2(curbackgroundframe, sig_start, sig_end, 
                                                     length, startbg2, totalcurbackground)

                    chr_list.append(chrom) #add chr to frame
                    start_list.append(sig_start) #add peak start to frame
                    end_list.append(sig_end) 

                    if record:
                    #add peak end to frame
                        center_list.append(peak_center) #add peak center to frame
                        num_TTAA_hops_list.append(num_TTAA_hops)
                        num_exp_hops_list.append(num_exp_hops)#add fraction of experiment hops in peak to frame
                        frac_exp_list.append(float(num_exp_hops)/total_experiment_hops)
                        tph_exp_list.append(float(num_exp_hops)*100000/total_experiment_hops)
                        num_bg_hops_list.append(num_bg_hops)
                        frac_bg_list.append(float(num_bg_hops)/total_background_hops)
                        tph_bg_list.append(float(num_bg_hops)*100000/total_background_hops)
                        tph_bgs = float(num_exp_hops)*100000/total_experiment_hops-float(num_bg_hops)*100000/total_background_hops
                        tph_bgs_list.append(tph_bgs)

                    # caluclate the final P value 

                    if lam_win_size == None:

                        if test_method == "poisson":

                            pvalue_list_TTAA.append(1-poisson.cdf((num_exp_hops+pseudocounts),
                                                                  lambdacurTTAA*num_TTAA_hops+pseudocounts))
                            pvalue_list_background.append(_compute_cumulative_poisson(num_exp_hops,
                                                                                     num_bg_hops,
                                                                                     totalcurChrom,
                                                                                     totalcurbackground,
                                                                                     pseudocounts))


                        elif test_method == "binomial":

                            pvalue_list_TTAA.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                              n=totalcurChrom, 
                                                              p=((num_TTAA_hops+pseudocounts)/totalcurTTAA) , 
                                                              alternative='greater').pvalue)
                            pvalue_list_background.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                                n=totalcurChrom, 
                                                                p=((num_bg_hops+pseudocounts)/totalcurbackground) , 
                                                                alternative='greater').pvalue)
                    else:

                        num_exp_hops_lam , starthoplam2= _findinsertionslen2(curChromnp, 
                                                                           sig_start - int(lam_win_size/2) +1,
                                                                            sig_end + int(lam_win_size/2) - 1, 
                                                                            length, starthoplam2, totalcurChrom)

                        num_exp_bg_lam, startbglam2 = _findinsertionslen2(curbackgroundframe, 
                                                             sig_start - int(lam_win_size/2) +1,
                                                             sig_end + int(lam_win_size/2) - 1,
                                                             length,startbglam2, totalcurbackground)

                        num_exp_TTAA_lam, startTTAAlam2 = _findinsertionslen2(curTTAAframe, 
                                                             sig_start - int(lam_win_size/2) +1,
                                                             sig_end + int(lam_win_size/2) - 1,
                                                             length, startTTAAlam2, totalcurTTAA)

                        if test_method == "poisson":

                            pvalue_list_TTAA.append(1-poisson.cdf((num_exp_hops+pseudocounts),
                                                                  (num_exp_hops_lam/num_exp_TTAA_lam)*num_TTAA_hops
                                                                  +pseudocounts))
                            pvalue_list_background.append(_compute_cumulative_poisson(num_exp_hops,
                                                                                     num_bg_hops,
                                                                                     num_exp_hops_lam,
                                                                                     num_exp_bg_lam,
                                                                                     pseudocounts))


                        elif test_method == "binomial":

                            pvalue_list_TTAA.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                              n=num_exp_hops_lam, 
                                                              p=((num_TTAA_hops+pseudocounts)/num_exp_TTAA_lam) , 
                                                              alternative='greater').pvalue)
                            if num_exp_bg_lam == 0:
                                pvalue_list_background.append(0)
                            else:
                                pvalue_list_background.append(binomtest(int(num_exp_hops+pseudocounts), 
                                                                n=num_exp_hops_lam, 
                                                                p=((num_bg_hops+pseudocounts)/num_exp_bg_lam) , 
                                                                alternative='greater').pvalue)



                    #number of hops that are a user-defined distance from peak center
                    sig_flag = 0


    if record:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End","Center",
                                              "Experiment Hops","Background Hops","Reference Hops",
                                              "pvalue Reference","pvalue Background"])


        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Hops"] = num_exp_hops_list 
        peaks_frame["Background Hops"] = num_bg_hops_list 
        peaks_frame["Reference Hops"] = num_TTAA_hops_list

        peaks_frame["Fraction Experiment"] = frac_exp_list 
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["Fraction background"] = frac_bg_list
        peaks_frame["TPH background"] = tph_bg_list
        peaks_frame["TPH background subtracted"] = tph_bgs_list


    else:
        peaks_frame = pd.DataFrame(columns = ["Chr","Start","End"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["pvalue Reference"] = pvalue_list_TTAA
    peaks_frame["pvalue Background"] = pvalue_list_background


    peaks_frame = peaks_frame[peaks_frame["pvalue Reference"] <= pvalue_cutoff_TTAA]
    peaks_frame = peaks_frame[peaks_frame["pvalue Background"] <= pvalue_cutoff_background]


    if record:
        return peaks_frame
    else:                    
        return peaks_frame[['Chr','Start','End']]
       
def _checkint(number,name):
    
    try:
        number = int(number)
    except:
        print('Please enter a valid positive number or 0 for' + name)
    if number <0 :
        raise ValueError('Please enter a valid positive number or 0 for' + name)
       
    return number

def _checkpvalue(number,name):
    
    try:
        number = float(number)
    except:
        print('Please enter a valid number (0,1) for ' + name)
    if number <0 or number >1 :
        raise ValueError('Please enter a valid number (0,1) for ' + name)
       
    return number

def _check_test_method(method):
    if method != "poisson" and  method != "binomial" :
        raise ValueError("Not valid a valid test method. Please input poisson or binomial.")

def callpeaks(
    expdata: pd.DataFrame, 
    background: Optional[pd.DataFrame] = None, 
    method: _Peakcalling_Method = "test", 
    reference: _reference = "hg38",
    pvalue_cutoff: float = 0.0001,  
    pvalue_cutoffbg: float = 0.0001, 
    pvalue_cutoffTTAA: float = 0.00001,
    min_hops: int = 5, 
    minlen: int = 0, 
    extend: int = 200, 
    maxbetween: int = 2000, 
    test_method: _PeaktestMethod = "poisson",
    window_size: int = 1500, 
    lam_win_size: Optional[int]  =100000, 
    step_size: int = 500, 
    pseudocounts: float = 0.2, 
    record: bool = True, 
    save: Optional[str] = None
    ) -> pd.DataFrame:
    
    
    """\
    Call peaks from ccf data.

    :param expdata:
        pd.DataFrame with first three columns as: chromosome, start, end.
    :param background:
        pd.DataFrame with first three columns as: chromosome, start, end. Default is `'None'` for the backgound free situation.
    :param method:
        The default method is `'test'`,
        `'test'` is a method considering the maxdistance between hops in the data,
        `'MACS2'` uses the idea adapted from [Zhang08]_,
        `here <https://hbctraining.github.io/Intro-to-ChIPseq/lessons/05_peak_calling_macs.html>`__,
        `Blockify` uses the method from `here <https://Blockify.readthedocs.io/en/latest/pages/introduction.html>`__.
    :param reference:
        Default method is `'hg38'`,
        We currently have `'hg38'` for human data, `'mm10'` for mouse data and `'yeast'` for yeast data.
    :param pvalue_cutoff:
        The P-value cutoff for backgound free situation. Default is 0.0001.
    :param pvalue_cutoffbg:
        The P-value cutoff for backgound data when backgound exists. Default is 0.0001.
    :param pvalue_cutoffTTAA:
        The P-value cutoff for reference data when backgound exists. Default is 0.00001. 
        Normally, pvalue_cutoffTTAA is recommended to be lower than pvalue_cutoffbg.
    :param min_hops:
        The number of minimal hops for each peak. Default is 5.
    :param minlen:
        Valid only for `'test'`. The minimal length for a peak without extend.  Default is 0.
    :param extend:
        Valid for `'test'` and `'MACS2'`. The length (bp) that peaks extend for both sides. Default is 200.
    :param maxbetween:
        Valid only for `'test'`. The maximum length of nearby hops within one peak. Default is 2000.
    :param test_method:
        The method for making hypothesis test. 
        We currently have `'poisson'` and `'binomial'` available. Default is `'poisson'`.
    :param window_size:
        Valid only for `'MACS2'`. The length of window we look for. Default is 1500.
    :param lam_win_size:
        Valid for `'test'` and `'MACS2'`. The length of peak area we consider when performing the test.
    :param step_size:
        Valid only for `'MACS2'`. The length of each step. Default is 500.
    :param pseudocounts:
        Number for pseudocounts added for the pyhothesis test. Default is 0.2.
    :param record:
        Whether to record other information or not. Default is `True`.
        If it is `False`, the output would only have three columns: Chromosome, Start, End.
    :param save:
        The file name for the file we save. Default is `'None'` and would not be saved.
       

    :Returns:
        | **Chr** - The chromosome of the peak. 
        | **Start** - The start point of the peak. 
        | **End** - The end point of the peak. 
        | **Experiment Hops** - The total number of hops within a peak in the experiment data.
        | **Reference Hops** - The total number of hops of within a peak in the reference data.
        | **Background Hops** - The total number of hops within a peak in the experiment data.
        | **Expected Hops** - The total number of expected hops under null hypothesis from the reference data (for background free situation). 
        | **Expected Hops background** - The total number of expected hops under null hypothesis from the background data (for background situation).
        | **Expected Hops Reference** - The total number of expected hops under null hypothesis from the reference data (for background situation).
        | **pvalue** - The pvalue we calculate from null hypothesis (for background free situation or Blockify).
        | **pvalue Reference** - The total number of hops of within a peak in the reference data (for background situation).
        | **pvalue Background** - The total number of hops of within a peak in the reference data (for background situation).
        | **Fraction Experiment** - 
        | **TPH Experiment** - 
        | **Fraction Background** - 
        | **TPH Background** - 
        | **TPH Background subtracted** - 

   
    :Examples:
    >>> import pycallingcards as cc
    >>> ccf_data = cc.datasets.mousecortex_ccf()
    >>> peak_data = cc.pp.callpeaks(ccf_data, method = "test", reference = "mm10",  maxbetween = 2000,pvalue_cutoff = 0.01, lam_win_size = 1000000,  pseudocounts = 1, record = True)
                  
    """

    if type(expdata) != pd.DataFrame :
        raise ValueError("Please input a pandas dataframe as the expression data.")
        
    if type(record) != bool:
        raise ValueError('Please enter a True/ False for record')
      
    if type(background) == pd.DataFrame :
        
        length = 3
        
        if method == "MACS2":
            
            print("For the MACS2 method with background, [expdata, background, reference, pvalue_cutoffbg, pvalue_cutoffTTAA, lam_win_size, window_size, step_size, extend, pseudocounts, test_method, min_hops, record] would be utilized.")
            
            if reference == "hg38":
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",delimiter="\t",header=None)
            elif reference == "mm10":
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",delimiter="\t",header=None)
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoffbg,"pvalue_cutoffbg")
            _checkpvalue(pvalue_cutoffTTAA,"pvalue_cutoffTTAA")
            
            window_size = _checkint(window_size,"window_size")
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size,"lam_win_size")
            extend = _checkint(extend,"extend")
            step_size = _checkint(step_size,"step_size")
            
                
            _check_test_method(test_method)
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
        
            if save == None:
                
                return _callpeaksMACS2new2(expdata, background, TTAAframe, length, extend = extend, lam_win_size = lam_win_size,
                      pvalue_cutoff_background =  pvalue_cutoffbg,  pvalue_cutoff_TTAA = pvalue_cutoffTTAA,
                      window_size = window_size, step_size = step_size, pseudocounts = pseudocounts,
                      test_method= test_method, min_hops = min_hops, record = record).reset_index(drop = True)
            else:
                
                data = _callpeaksMACS2new2(expdata, background, TTAAframe, length, extend = extend, lam_win_size = lam_win_size,
                      pvalue_cutoff_background =  pvalue_cutoffbg,  pvalue_cutoff_TTAA = pvalue_cutoffTTAA,
                      window_size = window_size, step_size = step_size, pseudocounts = pseudocounts,
                      test_method= test_method, min_hops = min_hops, record = record).reset_index(drop = True)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
                
        
        elif method == "test":
            
            print("For the test method with background, [expdata, background, reference, pvalue_cutoffbg, pvalue_cutoffTTAA, lam_win_size, pseudocounts, minlen, extend, maxbetween, test_method, min_hops, record] would be utilized.")
            
            if reference == "hg38":
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",delimiter="\t",header=None)
            elif reference == "mm10":
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",delimiter="\t",header=None)
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoffbg,"pvalue_cutoffbg")
            _checkpvalue(pvalue_cutoffTTAA,"pvalue_cutoffTTAA")
                
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size,"lam_win_size")
            extend = _checkint(extend,"extend")
            _checkint(pseudocounts,"pseudocounts")
            _check_test_method(test_method)
            
            minlen = _checkint(minlen,"minlen")
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
            maxbetween = _checkint(maxbetween,"maxbetween")
            
            _check_test_method(test_method)
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
            
            if save == None:
                
                return _test2(expdata, background, TTAAframe, length, pvalue_cutoffbg = pvalue_cutoffbg, 
                        pvalue_cutoffTTAA = pvalue_cutoffTTAA,  mininser = min_hops, minlen = minlen,
                        extend = extend, maxbetween = maxbetween,  lam_win_size = lam_win_size,  
                        pseudocounts = pseudocounts, test_method = test_method, record = record )
            else:
                
                data = _test2(expdata, background, TTAAframe, length, pvalue_cutoffbg = pvalue_cutoffbg, 
                        pvalue_cutoffTTAA = pvalue_cutoffTTAA,  mininser = min_hops, minlen = minlen,
                        extend = extend, maxbetween = maxbetween,  lam_win_size = lam_win_size,  
                        pseudocounts = pseudocounts, test_method = test_method, record = record )
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        elif method == "Blockify":
            
            print("For the Blockify method with background, [expdata, background, pvalue_cutoff, pseudocounts, test_method,  record] would be utilized.")
                
            if type(record) != bool:
                raise ValueError('Please enter a True/ False for record')
                
            _check_test_method(test_method)
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
            _checkint(pseudocounts,"pseudocounts")
            
            if save == None:
            
                return _Blockify(expdata, background, length, pvalue_cutoff = pvalue_cutoff, pseudocounts = pseudocounts,
                            test_method = test_method , record = record)
            else:
                
                data = _Blockify(expdata, background, length, pvalue_cutoff = pvalue_cutoff, pseudocounts = pseudocounts,
                            test_method = test_method , record = record)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        if method == "MACS2_old":
            
            print("For the MACS2 method with background, [expdata, background, reference, pvalue, lam_win_size, window_size, step_size,pseudocounts,  record] would be utilized.")
            
            if reference == "hg38":
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",delimiter="\t",header=None)
            elif reference == "mm10":
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",delimiter="\t",header=None)
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
            
            window_size = _checkint(window_size,"window_size")
            lam_win_size = _checkint(lam_win_size,"lam_win_size")
            step_size = _checkint(step_size,"step_size")
            
            if save == None:
            
                return _callpeaksMACS2(expdata, background, TTAAframe, length, window_size = window_size, 
                                      lam_win_size=lam_win_size, step_size = step_size,
                                  pseudocounts = pseudocounts ,pvalue_cutoff = pvalue_cutoff, record = record).reset_index(drop = True)
            else:
                
                data = _callpeaksMACS2(expdata, background, TTAAframe, length, window_size = window_size, 
                                      lam_win_size=lam_win_size, step_size = step_size,
                                  pseudocounts = pseudocounts ,pvalue_cutoff = pvalue_cutoff, record = record).reset_index(drop = True)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        else:
            
            raise ValueError("Not valid Method.")
            
            
    if background == None:
            
        
        if method == "MACS2":
            
            print("For the MACS2 method without background, [expdata, reference, pvalue_cutoff, lam_win_size, window_size, step_size, extend, pseudocounts, test_method, min_hops, record] would be utilized.")
            
            if reference == "hg38":
                
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "mm10":
                
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "yeast":
                
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/yeast_Background.ccf",delimiter="\t",header=None)
                length = 0
                
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
            
            window_size = _checkint(window_size,"window_size")
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size,"lam_win_size")
            extend = _checkint(extend,"extend")
            step_size = _checkint(step_size,"step_size")
            _checkint(pseudocounts,"pseudocounts")

            _check_test_method(test_method)
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
        
            if save == None:
                
                return _callpeaksMACS2_bfnew2(expdata, TTAAframe, length, extend = extend, 
                      pvalue_cutoff =  pvalue_cutoff, window_size = window_size, 
                      lam_win_size = lam_win_size,  step_size = step_size, pseudocounts = pseudocounts,
                      test_method= test_method, min_hops = min_hops, record = record).reset_index(drop = True)
            else:
                
                data = _callpeaksMACS2_bfnew2(expdata, TTAAframe, length, extend = extend, 
                      pvalue_cutoff =  pvalue_cutoff, window_size = window_size, 
                      lam_win_size = lam_win_size,  step_size = step_size, pseudocounts = pseudocounts,
                      test_method= test_method, min_hops = min_hops, record = record).reset_index(drop = True)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        elif method == "test":
            
            print("For the test method without background, [expdata, reference, pvalue_cutoff, lam_win_size, pseudocounts, minlen, extend, maxbetween, test_method, min_hops, record] would be utilized.")
            
            if reference == "hg38":
                
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "mm10":
                
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "yeast":
                
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/yeast_Background.ccf",delimiter="\t",header=None)
                length = 0
                
            else:
                raise ValueError("Not valid reference.")
                
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
                
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size,"lam_win_size")
            extend = _checkint(extend,"extend")
            _checkint(pseudocounts,"pseudocounts")
            _check_test_method(test_method)
            
            minlen = _checkint(minlen,"minlen")
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
            maxbetween = _checkint(maxbetween,"maxbetween")
            
            _check_test_method(test_method)
            min_hops = _checkint(min_hops,"min_hops")
            min_hops = max(min_hops,1)
            
        
            if save == None:
                
                return _test_bf2(expdata, TTAAframe, length, pvalue_cutoff = pvalue_cutoff, mininser = min_hops, minlen = minlen,
                        extend = extend, maxbetween = maxbetween,  lam_win_size = lam_win_size,  
                        pseudocounts = pseudocounts, test_method = test_method, record = record )
            else:
                
                data = _test_bf2(expdata, TTAAframe, length, pvalue_cutoff = pvalue_cutoff, mininser = min_hops, minlen = minlen,
                        extend = extend, maxbetween = maxbetween,  lam_win_size = lam_win_size,  
                        pseudocounts = pseudocounts, test_method = test_method, record = record )
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        elif method == "Blockify":
            
            print("For the Blockify method with background, [expdata, reference, pvalue_cutoff, pseudocounts, test_method,  record] would be utilized.")
            
            if reference == "hg38":
                
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "mm10":
                
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",delimiter="\t",header=None)
                length = 3
                
            elif reference == "yeast":
                
                TTAAframe = pd.read_csv("https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/yeast_Background.ccf",delimiter="\t",header=None)
                length = 0
                
            else:
                raise ValueError("Not valid reference.")
                
            _checkint(pseudocounts,"pseudocounts")

            if type(record) != bool:
                raise ValueError('Please enter a True/ False for record')
                
            _check_test_method(test_method)
            _checkpvalue(pvalue_cutoff,"pvalue_cutoff")
            
            if save == None:
                
                return  _Blockify(expdata, TTAAframe, length, pvalue_cutoff = pvalue_cutoff, pseudocounts = pseudocounts,
                            test_method = test_method , record = record)
            else:
                
                data =  _Blockify(expdata, TTAAframe, length, pvalue_cutoff = pvalue_cutoff, pseudocounts = pseudocounts,
                            test_method = test_method , record = record)
                
                data.to_csv(save,sep ="\t",header = None, index = None)
                
                return data
        
        
        else:
            
            raise ValueError("Not valid Method.")

    else :
        
        raise ValueError("Not a valid background.")