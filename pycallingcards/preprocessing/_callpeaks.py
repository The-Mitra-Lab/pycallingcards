from typing import Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import tqdm
from numba import jit

_Peakcalling_Method = Optional[Literal["CCcaller", "cc_tools", "Blockify"]]
_reference = Optional[Literal["hg38", "mm10", "sacCer3"]]
_PeakTestMethod = Optional[Literal["poisson", "binomial"]]


@jit(nopython=True)
def _findinsertionslen2(
    Chrom, start, end, length=3, startpoint=0, totallength=100000000
):
    # function to calculate the number of insertions in the spcific area of chromosomes
    count = 0
    initial = startpoint
    flag = 0

    for i in range(startpoint, totallength):
        if Chrom[i] >= start - length and Chrom[i] <= end:
            if flag == 0:
                initial = i
                flag = 1
            count += 1
        elif Chrom[i] > end and count != 0:
            return count, initial

    return count, initial


def _findinsertionslen(Chrom, start, end, length=3):

    # function to calculate the number of insertions in the spcific area of chromosomes
    return len(Chrom[(Chrom >= max(start - length, 0)) & (Chrom <= end)])


def _findinsertions(Chrom, start, end, length=3):

    # function returns of insertions in the spcific area of chromosomes
    return Chrom[(Chrom >= max(start - length, 0)) & (Chrom <= end)]


def _compute_cumulative_poisson(
    exp_insertions_region,
    bg_insertions_region,
    total_exp_insertions,
    total_bg_insertions,
    pseudocounts,
):

    from scipy.stats import poisson

    # Calculating the probability under the hypothesis of possion distribution
    if total_bg_insertions >= total_exp_insertions:
        return 1 - poisson.cdf(
            (exp_insertions_region + pseudocounts),
            bg_insertions_region
            * (float(total_exp_insertions) / float(total_bg_insertions))
            + pseudocounts,
        )
    else:
        return 1 - poisson.cdf(
            (
                (
                    exp_insertions_region
                    * (float(total_bg_insertions) / float(total_exp_insertions))
                )
                + pseudocounts
            ),
            bg_insertions_region + pseudocounts,
        )


def _CCcallerCompare_bf2(
    bound: list,
    curChromnp: np.ndarray,
    curframe: np.ndarray,
    length: int,
    lam_win_size: Optional[int],
    boundnew: list,
    pseudocounts: float = 0.2,
    pvalue_cutoff: float = 0.00001,
    chrom: str = None,
    test_method: _PeakTestMethod = "poisson",
    record: bool = True,
    minnum: int = 0,
) -> list:

    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":
        from scipy.stats import binom_test

    # CCcaller whether the potiential peaks are true peaks by comparing to other data

    startpointTTAA = 0

    if lam_win_size != None:
        startpointTTAAlam = 0
        startpointboundlam = 0

    totallengthcurframe = len(curframe)
    totallengthcurChromnp = len(curChromnp)

    if minnum == 0:

        for i in range(len(bound)):

            # calculate the total number of insertions in total
            TTAAnum, startpointTTAA = _findinsertionslen2(
                curframe,
                bound[i][0],
                bound[i][1],
                length,
                startpointTTAA,
                totallengthcurframe,
            )
            boundnum = bound[i][2]

            if lam_win_size == None:

                scaleFactor = float(totallengthcurChromnp / totallengthcurframe)
                lam = TTAAnum * scaleFactor + pseudocounts

                if test_method == "poisson":
                    pvalue = 1 - poisson.cdf(boundnum, lam)
                elif test_method == "binomial":
                    pvalue = binom_test(
                        int(boundnum + pseudocounts),
                        n=totallengthcurChromnp,
                        p=((TTAAnum + pseudocounts) / totallengthcurframe),
                        alternative="greater",
                    ).pvalue

            else:

                TTAAnumlam, startpointTTAAlam = _findinsertionslen2(
                    curframe,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startpointTTAAlam,
                    totallengthcurframe,
                )
                boundnumlam, startpointboundlam = _findinsertionslen2(
                    curChromnp,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startpointboundlam,
                    totallengthcurChromnp,
                )

                scaleFactor = float(boundnumlam / TTAAnumlam)
                lam = TTAAnum * scaleFactor + pseudocounts

                if test_method == "poisson":
                    pvalue = 1 - poisson.cdf(boundnum, lam)
                elif test_method == "binomial":
                    pvalue = binom_test(
                        int(boundnum + pseudocounts),
                        n=boundnumlam,
                        p=((TTAAnum + pseudocounts) / TTAAnumlam),
                        alternative="greater",
                    ).pvalue

            if pvalue <= pvalue_cutoff:
                if record:
                    boundnew.append(
                        [
                            chrom,
                            bound[i][0],
                            bound[i][1],
                            boundnum,
                            TTAAnum,
                            lam,
                            pvalue,
                        ]
                    )
                else:
                    boundnew.append([chrom, bound[i][0], bound[i][1]])

        return boundnew

    else:

        startchrom = 0

        for i in range(len(bound)):

            # calculate the total number of insertions in total
            TTAAnum, startpointTTAA = _findinsertionslen2(
                curframe,
                bound[i][0],
                bound[i][1],
                length,
                startpointTTAA,
                totallengthcurframe,
            )

            boundnum, startchrom = _findinsertionslen2(
                curChromnp,
                bound[i][0],
                bound[i][1],
                length,
                startchrom,
                totallengthcurChromnp,
            )

            if lam_win_size == None:

                scaleFactor = float(totallengthcurChromnp / totallengthcurframe)
                lam = TTAAnum * scaleFactor + pseudocounts

                if test_method == "poisson":
                    pvalue = 1 - poisson.cdf(boundnum, lam)
                elif test_method == "binomial":
                    pvalue = binom_test(
                        int(boundnum + pseudocounts),
                        n=totallengthcurChromnp,
                        p=((TTAAnum + pseudocounts) / totallengthcurframe),
                        alternative="greater",
                    ).pvalue

            else:

                TTAAnumlam, startpointTTAAlam = _findinsertionslen2(
                    curframe,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startpointTTAAlam,
                    totallengthcurframe,
                )
                boundnumlam, startpointboundlam = _findinsertionslen2(
                    curChromnp,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startpointboundlam,
                    totallengthcurChromnp,
                )

                scaleFactor = float(boundnumlam / TTAAnumlam)
                lam = TTAAnum * scaleFactor + pseudocounts

                if test_method == "poisson":
                    pvalue = 1 - poisson.cdf(boundnum, lam)
                elif test_method == "binomial":
                    pvalue = binom_test(
                        int(boundnum + pseudocounts),
                        n=boundnumlam,
                        p=((TTAAnum + pseudocounts) / TTAAnumlam),
                        alternative="greater",
                    ).pvalue

            if pvalue <= pvalue_cutoff:
                if record:
                    boundnew.append(
                        [
                            chrom,
                            bound[i][0],
                            bound[i][1],
                            boundnum,
                            TTAAnum,
                            lam,
                            pvalue,
                        ]
                    )
                else:
                    boundnew.append([chrom, bound[i][0], bound[i][1]])

        return boundnew


def _CCcallerCompare2(
    bound: list,
    curChromnp: np.ndarray,
    curbgframe: np.ndarray,
    curTTAAframenp: np.ndarray,
    length: int,
    lam_win_size: Optional[int],
    boundnew: list,
    pseudocounts: float,
    pvalue_cutoffbg: float,
    pvalue_cutoffTTAA: float,
    chrom: str,
    test_method: _PeakTestMethod,
    record: bool,
    minnum: int,
) -> list:

    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":
        from scipy.stats import binom_test

    # CCcaller whether the potiential peaks are true peaks by comparing to other data

    startbg = 0
    startTTAA = 0

    totalcurChrom = len(curChromnp)
    totalcurbackground = len(curbgframe)
    totalcurTTAA = len(curTTAAframenp)

    if lam_win_size != None:

        startbglam = 0
        startTTAAlam = 0
        startboundlam = 0

    if minnum == 0:

        for i in range(len(bound)):

            bgnum, startbg = _findinsertionslen2(
                curbgframe,
                bound[i][0],
                bound[i][1],
                length,
                startbg,
                totalcurbackground,
            )
            TTAAnum, startTTAA = _findinsertionslen2(
                curTTAAframenp,
                bound[i][0],
                bound[i][1],
                length,
                startTTAA,
                totalcurTTAA,
            )

            boundnum = bound[i][2]

            if lam_win_size == None:

                scaleFactorTTAA = totalcurChrom / totalcurTTAA
                lamTTAA = TTAAnum * scaleFactorTTAA + pseudocounts

                scaleFactorbg = totalcurChrom / totalcurbackground
                lambg = bgnum * scaleFactorbg + pseudocounts

                if test_method == "poisson":

                    pvalueTTAA = 1 - poisson.cdf(boundnum, lamTTAA)
                    pvaluebg = _compute_cumulative_poisson(
                        boundnum, bgnum, totalcurChrom, totalcurbackground, pseudocounts
                    )

                elif test_method == "binomial":

                    pvalueTTAA = binom_test(
                        int(boundnum + pseudocounts),
                        n=totalcurChrom,
                        p=((TTAAnum + pseudocounts) / totalcurTTAA),
                        alternative="greater",
                    ).pvalue
                    pvaluebg = binom_test(
                        int(boundnum + pseudocounts),
                        n=totalcurChrom,
                        p=((bgnum + pseudocounts) / totalcurbackground),
                        alternative="greater",
                    ).pvalue

            else:

                bgnumlam, startbglam = _findinsertionslen2(
                    curbgframe,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startbglam,
                    totalcurbackground,
                )
                TTAAnumlam, startTTAAlam = _findinsertionslen2(
                    curTTAAframenp,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startTTAAlam,
                    totalcurTTAA,
                )
                boundnumlam, startboundlam = _findinsertionslen2(
                    curChromnp,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startboundlam,
                    totalcurChrom,
                )

                scaleFactorTTAA = boundnumlam / TTAAnumlam
                lamTTAA = TTAAnum * scaleFactorTTAA + pseudocounts

                if bgnumlam != 0:
                    scaleFactorbg = boundnumlam / bgnumlam
                    lambg = bgnum * scaleFactorbg + pseudocounts
                else:
                    lambg = 0

                if test_method == "poisson":

                    pvalueTTAA = 1 - poisson.cdf(boundnum, lamTTAA)
                    pvaluebg = _compute_cumulative_poisson(
                        boundnum, bgnum, boundnumlam, bgnumlam, pseudocounts
                    )

                elif test_method == "binomial":

                    pvalueTTAA = binom_test(
                        int(boundnum + pseudocounts),
                        n=boundnumlam,
                        p=((TTAAnum + pseudocounts) / TTAAnumlam),
                        alternative="greater",
                    ).pvalue

                    if bgnumlam == 0:
                        pvaluebg = 0
                    else:
                        pvaluebg = binom_test(
                            int(boundnum + pseudocounts),
                            n=boundnumlam,
                            p=((bgnum + pseudocounts) / bgnumlam),
                            alternative="greater",
                        ).pvalue

            if pvaluebg <= pvalue_cutoffbg and pvalueTTAA <= pvalue_cutoffTTAA:

                if record:
                    boundnew.append(
                        [
                            chrom,
                            bound[i][0],
                            bound[i][1],
                            boundnum,
                            bgnum,
                            TTAAnum,
                            lambg,
                            lamTTAA,
                            pvaluebg,
                            pvalueTTAA,
                        ]
                    )
                else:
                    boundnew.append([chrom, bound[i][0], bound[i][1]])

        return boundnew

    else:

        startchrom = 0

        for i in range(len(bound)):

            bgnum, startbg = _findinsertionslen2(
                curbgframe,
                bound[i][0],
                bound[i][1],
                length,
                startbg,
                totalcurbackground,
            )
            TTAAnum, startTTAA = _findinsertionslen2(
                curTTAAframenp,
                bound[i][0],
                bound[i][1],
                length,
                startTTAA,
                totalcurTTAA,
            )

            boundnum, startchrom = _findinsertionslen2(
                curChromnp, bound[i][0], bound[i][1], length, startchrom, totalcurChrom
            )

            if lam_win_size == None:

                scaleFactorTTAA = totalcurChrom / totalcurTTAA
                lamTTAA = TTAAnum * scaleFactorTTAA + pseudocounts

                scaleFactorbg = totalcurChrom / totalcurbackground
                lambg = bgnum * scaleFactorbg + pseudocounts

                if test_method == "poisson":

                    pvalueTTAA = 1 - poisson.cdf(boundnum, lamTTAA)
                    pvaluebg = _compute_cumulative_poisson(
                        boundnum, bgnum, totalcurChrom, totalcurbackground, pseudocounts
                    )

                elif test_method == "binomial":

                    pvalueTTAA = binom_test(
                        int(boundnum + pseudocounts),
                        n=totalcurChrom,
                        p=((TTAAnum + pseudocounts) / totalcurTTAA),
                        alternative="greater",
                    ).pvalue
                    pvaluebg = binom_test(
                        int(boundnum + pseudocounts),
                        n=totalcurChrom,
                        p=((bgnum + pseudocounts) / totalcurbackground),
                        alternative="greater",
                    ).pvalue

            else:

                bgnumlam, startbglam = _findinsertionslen2(
                    curbgframe,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startbglam,
                    totalcurbackground,
                )
                TTAAnumlam, startTTAAlam = _findinsertionslen2(
                    curTTAAframenp,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startTTAAlam,
                    totalcurTTAA,
                )
                boundnumlam, startboundlam = _findinsertionslen2(
                    curChromnp,
                    bound[i][0] - lam_win_size / 2 + 1,
                    bound[i][1] + lam_win_size / 2,
                    length,
                    startboundlam,
                    totalcurChrom,
                )

                scaleFactorTTAA = boundnumlam / TTAAnumlam
                lamTTAA = TTAAnum * scaleFactorTTAA + pseudocounts

                if bgnumlam != 0:
                    scaleFactorbg = boundnumlam / bgnumlam
                    lambg = bgnum * scaleFactorbg + pseudocounts
                else:
                    lambg = 0

                if test_method == "poisson":

                    pvalueTTAA = 1 - poisson.cdf(boundnum, lamTTAA)
                    pvaluebg = _compute_cumulative_poisson(
                        boundnum, bgnum, boundnumlam, bgnumlam, pseudocounts
                    )

                elif test_method == "binomial":

                    pvalueTTAA = binom_test(
                        int(boundnum + pseudocounts),
                        n=boundnumlam,
                        p=((TTAAnum + pseudocounts) / TTAAnumlam),
                        alternative="greater",
                    ).pvalue

                    if bgnumlam == 0:
                        pvaluebg = 0
                    else:
                        pvaluebg = binom_test(
                            int(boundnum + pseudocounts),
                            n=boundnumlam,
                            p=((bgnum + pseudocounts) / bgnumlam),
                            alternative="greater",
                        ).pvalue

            if pvaluebg <= pvalue_cutoffbg and pvalueTTAA <= pvalue_cutoffTTAA:

                if record:
                    boundnew.append(
                        [
                            chrom,
                            bound[i][0],
                            bound[i][1],
                            boundnum,
                            bgnum,
                            TTAAnum,
                            lambg,
                            lamTTAA,
                            pvaluebg,
                            pvalueTTAA,
                        ]
                    )
                else:
                    boundnew.append([chrom, bound[i][0], bound[i][1]])

        return boundnew


def _CCcaller_bf2(
    expdata: pd.DataFrame,
    TTAAframe: pd.DataFrame,
    length: int,
    pvalue_cutoff: float = 0.01,
    mininser: int = 5,
    minlen: int = 0,
    extend: int = 150,
    maxbetween: int = 2800,
    lam_win_size: Optional[int] = None,
    pseudocounts: float = 0.2,
    test_method: _PeakTestMethod = "poisson",
    record: bool = False,
    minnum: int = 0,
) -> pd.DataFrame:

    # The chromosomes we need to consider
    chrm = list(expdata["Chr"].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):

        curTTAAframe = np.array(list(TTAAframe[TTAAframe["Chr"] == chrom]["Start"]))
        if len(curTTAAframe) == 0:
            continue

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata["Chr"] == chrom]["Start"])
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()

        # make a summary of our current insertion start points
        unique, counts = np.unique(curChromnp, return_counts=True)
        counts = np.concatenate((counts, np.array([0] * 2)))

        # create a list to find out the protintial peak region of their bounds
        bound = []

        # initial the start point, end point and the totol number of insertions
        startbound = 0
        endbound = 0
        insertionbound = 0

        # calculate the distance between each points
        dif1 = np.diff(unique, axis=0)
        # add a zero to help end the following loop at the end
        dif1 = np.concatenate((dif1, np.array([maxbetween + 1] * 2)))

        # look for the uique insertion points
        for i in range(len(unique)):
            if startbound == 0:
                startbound = unique[i]
                insertionbound += counts[i]

                if (dif1[i] > maxbetween) or (
                    counts[(i + 1)] <= minnum
                    and (
                        dif1[i + 1] > float(maxbetween) / 3 or counts[(i + 2)] <= minnum
                    )
                ):
                    endbound = unique[i]
                    if (insertionbound >= mininser) and (
                        (endbound - startbound) >= minlen
                    ):
                        bound.append(
                            [
                                max(startbound - extend, 0),
                                endbound + 4 + extend,
                                insertionbound,
                                endbound - startbound,
                            ]
                        )
                    startbound = 0
                    endbound = 0
                    insertionbound = 0
            else:
                insertionbound += counts[i]
                if (dif1[i] > maxbetween) or (
                    counts[(i + 1)] <= minnum
                    and (
                        dif1[i + 1] > float(maxbetween) / 3 or counts[(i + 2)] <= minnum
                    )
                ):
                    endbound = unique[i]
                    if (insertionbound >= mininser) and (
                        (endbound - startbound) >= minlen
                    ):
                        bound.append(
                            [
                                max(startbound - extend, 0),
                                endbound + 4 + extend,
                                insertionbound,
                                endbound - startbound,
                            ]
                        )
                    startbound = 0
                    endbound = 0
                    insertionbound = 0

        boundnew = _CCcallerCompare_bf2(
            bound,
            curChromnp,
            curTTAAframe,
            length,
            lam_win_size,
            boundnew,
            pseudocounts,
            pvalue_cutoff,
            chrom,
            test_method=test_method,
            record=record,
            minnum=minnum,
        )

    for inser_num in range(len(bound) - 1):
        if boundnew[inser_num + 1][1] < boundnew[inser_num][2]:
            boundnew[inser_num + 1][1] = boundnew[inser_num][2]

    if record:
        return pd.DataFrame(
            boundnew,
            columns=[
                "Chr",
                "Start",
                "End",
                "Experiment Insertions",
                "Reference Insertions",
                "Expected Insertions",
                "pvalue",
            ],
        )

    else:
        # print(boundnew)
        return pd.DataFrame(boundnew, columns=["Chr", "Start", "End"])


def _CCcaller2(
    expdata: pd.DataFrame,
    backgroundframe: pd.DataFrame,
    TTAAframe: pd.DataFrame,
    length: int,
    pvalue_cutoffbg: float = 0.00001,
    pvalue_cutoffTTAA: float = 0.000001,
    mininser: int = 5,
    minlen: int = 0,
    extend: int = 150,
    maxbetween: int = 2800,
    lam_win_size: Optional[int] = None,
    pseudocounts: float = 0.2,
    test_method: _PeakTestMethod = "poisson",
    record: bool = False,
    minnum: int = 0,
) -> pd.DataFrame:

    # The chromosomes we need to consider
    chrm = list(expdata["Chr"].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):

        curbackgroundframe = np.array(
            list(backgroundframe[backgroundframe["Chr"] == chrom]["Start"])
        )
        if len(curbackgroundframe) == 0:
            continue
        curbackgroundframe.sort()

        curTTAAframe = np.array(list(TTAAframe[TTAAframe["Chr"] == chrom]["Start"]))
        if len(curTTAAframe) == 0:
            continue

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata["Chr"] == chrom]["Start"])
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()

        # make a summary of our current insertion start points
        unique, counts = np.unique(np.array(curChrom), return_counts=True)
        counts = np.concatenate((counts, np.array(2 * [0])))

        # create a list to find out the protintial peak region of their bounds
        bound = []

        # initial the start point, end point and the totol number of insertions
        startbound = 0
        endbound = 0
        insertionbound = 0

        # calculate the distance between each points
        dif1 = np.diff(unique, axis=0)
        # add a zero to help end the following loop at the end
        dif1 = np.concatenate((dif1, np.array(2 * [maxbetween + 1])))

        # look for the uique insertion points
        for i in range(len(unique)):
            if startbound == 0:
                startbound = unique[i]
                insertionbound += counts[i]
                if (dif1[i] > maxbetween) or (
                    counts[(i + 1)] <= minnum
                    and (
                        dif1[i + 1] > float(maxbetween) / 3 or counts[(i + 2)] <= minnum
                    )
                ):
                    endbound = unique[i]
                    if (insertionbound >= mininser) and (
                        (endbound - startbound) >= minlen
                    ):
                        bound.append(
                            [
                                max(startbound - extend, 0),
                                endbound + 4 + extend,
                                insertionbound,
                                endbound - startbound,
                            ]
                        )
                    startbound = 0
                    endbound = 0
                    insertionbound = 0
            else:
                insertionbound += counts[i]
                if (dif1[i] > maxbetween) or (
                    counts[(i + 1)] <= minnum
                    and (
                        dif1[i + 1] > float(maxbetween) / 3 or counts[(i + 2)] <= minnum
                    )
                ):
                    endbound = unique[i]
                    if (insertionbound >= mininser) and (
                        (endbound - startbound) >= minlen
                    ):
                        bound.append(
                            [
                                max(startbound - extend, 0),
                                endbound + 4 + extend,
                                insertionbound,
                                endbound - startbound,
                            ]
                        )
                    startbound = 0
                    endbound = 0
                    insertionbound = 0

        boundnew = _CCcallerCompare2(
            bound,
            curChromnp,
            curbackgroundframe,
            curTTAAframe,
            length,
            lam_win_size,
            boundnew,
            pseudocounts,
            pvalue_cutoffbg,
            pvalue_cutoffTTAA,
            chrom,
            test_method,
            record,
            minnum,
        )

    for inser_num in range(len(bound) - 1):
        if boundnew[inser_num + 1][1] < boundnew[inser_num][2]:
            boundnew[inser_num + 1][1] = boundnew[inser_num][2]

    if record:
        return pd.DataFrame(
            boundnew,
            columns=[
                "Chr",
                "Start",
                "End",
                "Experiment Insertions",
                "Background insertions",
                "Reference Insertions",
                "Expected Insertions background",
                "Expected Insertions Reference",
                "pvalue Background",
                "pvalue Reference",
            ],
        )

    else:
        return pd.DataFrame(boundnew, columns=["Chr", "Start", "End"])


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
    test_method: _PeakTestMethod = "poisson",
    record: bool = True,
) -> list:
    # CCcaller whether the potiential peaks are true peaks by comparing to TTAAs

    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":
        from scipy.stats import binomCCcall

    last = -1
    Chrnumtotal = 0
    TTAAnumtotal = 0
    currentlen = len(curframe)

    for i in range(len(bound)):

        TTAAnum = bound[i][4]
        boundnum = bound[i][2]

        if test_method == "poisson":
            pValue = 1 - poisson.cdf(boundnum - 1, TTAAnum * scaleFactor + pseudocounts)

        elif test_method == "binomial":
            pValue = binom_test(
                int(boundnum + pseudocounts),
                n=len(curChrom),
                p=((TTAAnum + pseudocounts) / currentlen),
                alternative="greater",
            ).pvalue

        if pValue <= pvalue_cutoff and last == -1:

            last = i
            Chrnumtotal += boundnum
            TTAAnumtotal += TTAAnum

        elif pValue <= pvalue_cutoff and last != -1:

            Chrnumtotal += boundnum
            TTAAnumtotal += TTAAnum

        elif pValue > pvalue_cutoff and last != -1:

            if record:

                if test_method == "poisson":
                    pvalue = 1 - poisson.cdf(
                        Chrnumtotal - 1, TTAAnumtotal * scaleFactor + pseudocounts
                    )

                elif test_method == "binomial":
                    pvalue = binom_test(
                        int(Chrnumtotal + pseudocounts),
                        n=len(curChrom),
                        p=((TTAAnumtotal + pseudocounts) / currentlen),
                        alternative="greater",
                    ).pvalue

                boundnew.append(
                    [
                        chrom,
                        bound[last][0],
                        bound[i - 1][1],
                        Chrnumtotal,
                        TTAAnumtotal,
                        TTAAnumtotal * scaleFactor + pseudocounts,
                        pvalue,
                    ]
                )
            else:
                boundnew.append([chrom, bound[last][0], bound[i - 1][1]])

            last = -1
            Chrnumtotal = 0
            TTAAnumtotal = 0

    if last != -1:

        if record:

            if test_method == "poisson":
                pvalue = 1 - poisson.cdf(
                    Chrnumtotal - 1, TTAAnumtotal * scaleFactor + pseudocounts
                )

            elif test_method == "binomial":
                pvalue = binom_test(
                    int(Chrnumtotal + pseudocounts),
                    n=len(curChrom),
                    p=((TTAAnumtotal + pseudocounts) / currentlen),
                    alternative="greater",
                ).pvalue

            boundnew.append(
                [
                    chrom,
                    bound[last][0],
                    bound[i][1],
                    Chrnumtotal,
                    TTAAnumtotal,
                    TTAAnumtotal * scaleFactor + pseudocounts,
                    pvalue,
                ]
            )
        else:
            boundnew.append([chrom, bound[last][0], bound[i][1]])

    return boundnew


def _Blockify(
    expdata: pd.DataFrame,
    TTAAframe: pd.DataFrame,
    length: int,
    pvalue_cutoff: float = 0.0001,
    pseudocounts: float = 0.2,
    test_method: _PeakTestMethod = "poisson",
    record: bool = True,
) -> pd.DataFrame:

    from pybedtools import BedTool

    from . import _blo_segmentation as segmentation

    blo = segmentation.segment(
        BedTool.from_dataframe(expdata), "PELT", p0=0.05, prior=None
    ).blocks.to_dataframe()

    # The chromosomes we need to consider
    chrm = list(expdata["Chr"].unique())

    # create a list to record every peaks
    boundnew = []

    # going through one chromosome from another
    for chrom in tqdm.tqdm(chrm):

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata["Chr"] == chrom]["Start"])
        curChromnp = np.array(curChrom)

        curTTAAframe = np.array(list(TTAAframe[TTAAframe["Chr"] == chrom]["Start"]))

        if len(curTTAAframe) == 0:
            continue

        # make a summary of our current insertion start points
        unique, counts = np.unique(curChromnp, return_counts=True)

        # create a list to find out the protintial peak region of their bounds
        bound = []

        hist, bin_edges = np.histogram(
            curChromnp,
            bins=np.array(
                list(blo[blo["chrom"] == chrom]["start"])
                + [list(blo[blo["chrom"] == chrom]["end"])[-1]]
            ),
        )

        hist_TTAA, _ = np.histogram(
            curTTAAframe,
            bins=np.array(
                list(blo[blo["chrom"] == chrom]["start"])
                + [list(blo[blo["chrom"] == chrom]["end"])[-1]]
            ),
        )

        hist = list(hist)
        bin_edges = list(bin_edges.astype(int))
        for bins in range(len(bin_edges) - 1):
            bound.append(
                [
                    bin_edges[bins],
                    bin_edges[bins + 1],
                    hist[bins],
                    bin_edges[bins + 1] - bin_edges[bins],
                    hist_TTAA[bins],
                ]
            )

        boundnew = _BlockifyCompare(
            bound,
            curChromnp,
            curTTAAframe,
            length,
            boundnew,
            scaleFactor=len(curChromnp) / len(curTTAAframe),
            pseudocounts=pseudocounts,
            pvalue_cutoff=pvalue_cutoff,
            chrom=chrom,
            test_method=test_method,
            record=record,
        )

    if record:
        return pd.DataFrame(
            boundnew,
            columns=[
                "Chr",
                "Start",
                "End",
                "Experiment Insertions",
                "Reference Insertions",
                "Expected Insertions",
                "pvalue",
            ],
        )

    else:
        return pd.DataFrame(boundnew, columns=["Chr", "Start", "End"])


def _callpeakscc_tools(
    expdata: pd.DataFrame,
    background: pd.DataFrame,
    TTAAframe: pd.DataFrame,
    length: int,
    window_size: int = 1000,
    lam_win_size: Optional[int] = 100000,
    step_size: int = 500,
    pseudocounts: float = 0.2,
    pvalue_cutoff: float = 0.01,
    record: bool = False,
) -> pd.DataFrame:

    # function for cc_tools with background
    from scipy.stats import poisson

    # The chromosomes we need to consider
    chrm = list(expdata["Chr"].unique())

    chr_list = []
    start_list = []
    end_list = []
    list_of_l_names = ["bg", "1k", "5k", "10k"]
    pvalue_list = []

    if record:
        center_list = []
        num_exp_insertions_list = []
        num_bg_insertions_list = []
        frac_exp_list = []
        tph_exp_list = []
        frac_bg_list = []
        tph_bg_list = []
        tph_bgs_list = []
        lambda_type_list = []
        lambda_list = []
        lambda_insertion_list = []

    total_experiment_insertions = len(expdata)
    total_background_insertions = len(background)

    for chrom in tqdm.tqdm(chrm):

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata["Chr"] == chrom]["Start"])
        curChromnp = np.array(curChrom)
        # sort it so that we could accelarate the searching afterwards
        curChrom.sort()

        max_pos = curChrom[-1] + 4
        sig_start = 0
        sig_end = 0
        sig_flag = 0

        curTTAAframe = np.array(list(TTAAframe[TTAAframe["Chr"] == chrom]["Start"]))

        curbackgroundframe = np.array(
            list(background[background["Chr"] == chrom]["Start"])
        )

        sig_start = 0
        sig_end = 0
        sig_flag = 0

        for window_start in range(0, int(max_pos + window_size), int(step_size)):

            num_exp_insertions = _findinsertionslen(
                curChromnp, window_start, window_start + window_size - 1, length
            )
            if num_exp_insertions > 1:
                num_bg_insertions = _findinsertionslen(
                    curbackgroundframe,
                    window_start,
                    window_start + window_size - 1,
                    length,
                )
                p = _compute_cumulative_poisson(
                    num_exp_insertions,
                    num_bg_insertions,
                    total_experiment_insertions,
                    total_background_insertions,
                    pseudocounts,
                )
            else:
                p = 1

            # is this window significant?
            if p <= pvalue_cutoff:
                # was last window significant?
                if sig_flag:
                    # if so, extend end of windows
                    sig_end = window_start + window_size - 1
                else:
                    # otherwise, define new start and end and set flag
                    sig_start = window_start
                    sig_end = window_start + window_size - 1
                    sig_flag = 1

            else:
                # current window not significant.  Was last window significant?
                if sig_flag:

                    # add full sig window to the frame of peaks
                    # add chr, peak start, peak end
                    chr_list.append(chrom)  # add chr to frame
                    start_list.append(max(0, sig_start))  # add peak start to frame
                    end_list.append(sig_end)  # add peak end to frame

                    # compute peak center and add to frame
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    peak_center = np.median(overlap)

                    # add number of Experiment Insertions in peak to frame
                    num_exp_insertions = len(overlap)

                    # add number of background insertions in peak to frame
                    num_bg_insertions = _findinsertionslen(
                        curbackgroundframe, sig_start, sig_end, length
                    )

                    if record:
                        center_list.append(peak_center)  # add peak center to frame
                        num_exp_insertions_list.append(num_exp_insertions)
                        # add fraction of Experiment Insertions in peak to frame
                        frac_exp_list.append(
                            float(num_exp_insertions) / total_experiment_insertions
                        )
                        tph_exp_list.append(
                            float(num_exp_insertions)
                            * 100000
                            / total_experiment_insertions
                        )
                        num_bg_insertions_list.append(num_bg_insertions)
                        frac_bg_list.append(
                            float(num_bg_insertions) / total_background_insertions
                        )
                        tph_bg_list.append(
                            float(num_bg_insertions)
                            * 100000
                            / total_background_insertions
                        )

                    # find lambda and compute significance of peak
                    if (
                        total_background_insertions >= total_experiment_insertions
                    ):  # scale bg insertions down
                        # compute lambda bg
                        num_TTAAs = _findinsertionslen(
                            curTTAAframe, sig_start, sig_end, length
                        )
                        lambda_bg = (
                            num_bg_insertions
                            * (
                                float(total_experiment_insertions)
                                / total_background_insertions
                            )
                        ) / max(num_TTAAs, 1)

                        # compute lambda 1k
                        num_bg_insertions_1k = _findinsertionslen(
                            curbackgroundframe,
                            peak_center - 499,
                            peak_center + 500,
                            length,
                        )
                        num_TTAAs_1k = _findinsertionslen(
                            curTTAAframe, peak_center - 499, peak_center + 500, length
                        )
                        lambda_1k = (
                            num_bg_insertions_1k
                            * (
                                float(total_experiment_insertions)
                                / total_background_insertions
                            )
                        ) / (max(num_TTAAs_1k, 1))

                        # compute lambda 5k
                        num_bg_insertions_5k = _findinsertionslen(
                            curbackgroundframe,
                            peak_center - 2499,
                            peak_center + 2500,
                            length,
                        )
                        num_TTAAs_5k = _findinsertionslen(
                            curTTAAframe, peak_center - 2499, peak_center + 2500, length
                        )
                        lambda_5k = (
                            num_bg_insertions_5k
                            * (
                                float(total_experiment_insertions)
                                / total_background_insertions
                            )
                        ) / (max(num_TTAAs_5k, 1))

                        # compute lambda 10k
                        num_bg_insertions_10k = _findinsertionslen(
                            curbackgroundframe,
                            peak_center - 4999,
                            peak_center + 5000,
                            length,
                        )
                        num_TTAAs_10k = _findinsertionslen(
                            curTTAAframe, peak_center - 4999, peak_center + 5000, length
                        )
                        lambda_10k = (
                            num_bg_insertions_10k
                            * (
                                float(total_experiment_insertions)
                                / total_background_insertions
                            )
                        ) / (max(num_TTAAs_10k, 1))
                        lambda_f = max([lambda_bg, lambda_1k, lambda_5k, lambda_10k])

                        # record type of lambda used
                        index = [lambda_bg, lambda_1k, lambda_5k, lambda_10k].index(
                            max([lambda_bg, lambda_1k, lambda_5k, lambda_10k])
                        )
                        lambda_type_list.append(list_of_l_names[index])
                        # record lambda
                        lambda_list.append(lambda_f)
                        # compute pvalue and record it

                        pvalue = 1 - poisson.cdf(
                            (num_exp_insertions + pseudocounts),
                            lambda_f * max(num_TTAAs, 1) + pseudocounts,
                        )
                        pvalue_list.append(pvalue)

                        tph_bgs = (
                            float(num_exp_insertions)
                            * 100000
                            / total_experiment_insertions
                            - float(num_bg_insertions)
                            * 100000
                            / total_background_insertions
                        )

                        if record:
                            lambda_type_list.append(list_of_l_names[index])
                            lambda_list.append(lambda_f)
                            tph_bgs_list.append(tph_bgs)
                            lambda_insertion_list.append(lambda_f * max(num_TTAAs, 1))

                        index = [lambda_bg, lambda_1k, lambda_5k, lambda_10k].index(
                            max([lambda_bg, lambda_1k, lambda_5k, lambda_10k])
                        )
                        lambdatype = list_of_l_names[index]
                        # l = [pvalue,tph_bgs,lambda_f,lambdatype]

                    else:  # scale Experiment Insertions down

                        # compute lambda bg
                        num_TTAAs = _findinsertionslen(
                            curTTAAframe, sig_start, sig_end, length
                        )
                        lambda_bg = float(num_bg_insertions) / max(num_TTAAs, 1)

                        # compute lambda 1k
                        num_bg_insertions_1k = _findinsertionslen(
                            curbackgroundframe,
                            peak_center - 499,
                            peak_center + 500,
                            length,
                        )
                        num_TTAAs_1k = _findinsertionslen(
                            curTTAAframe, peak_center - 499, peak_center + 500, length
                        )
                        lambda_1k = float(num_bg_insertions_1k) / (max(num_TTAAs_1k, 1))

                        # compute lambda 5k
                        num_bg_insertions_5k = _findinsertionslen(
                            curbackgroundframe,
                            peak_center - 2499,
                            peak_center + 2500,
                            length,
                        )
                        num_TTAAs_5k = _findinsertionslen(
                            curTTAAframe, peak_center - 2499, peak_center + 2500, length
                        )
                        lambda_5k = float(num_bg_insertions_5k) / (max(num_TTAAs_5k, 1))

                        # compute lambda 10k
                        num_bg_insertions_10k = _findinsertionslen(
                            curbackgroundframe,
                            peak_center - 4999,
                            peak_center + 5000,
                            length,
                        )
                        num_TTAAs_10k = _findinsertionslen(
                            curTTAAframe, peak_center - 4999, peak_center + 5000, length
                        )
                        lambda_10k = float(num_bg_insertions_10k) / (
                            max(num_TTAAs_10k, 1)
                        )
                        lambda_f = max([lambda_bg, lambda_1k, lambda_5k, lambda_10k])

                        # record type of lambda used
                        index = [lambda_bg, lambda_1k, lambda_5k, lambda_10k].index(
                            max([lambda_bg, lambda_1k, lambda_5k, lambda_10k])
                        )

                        # compute pvalue and record it
                        pvalue = 1 - poisson.cdf(
                            (
                                (
                                    float(total_background_insertions)
                                    / total_experiment_insertions
                                )
                                * num_exp_insertions
                                + pseudocounts
                            ),
                            lambda_f * max(num_TTAAs, 1) + pseudocounts,
                        )
                        pvalue_list.append(pvalue)

                        tph_bgs = (
                            float(num_exp_insertions)
                            * 100000
                            / total_experiment_insertions
                            - float(num_bg_insertions)
                            * 100000
                            / total_background_insertions
                        )

                        if record:
                            lambda_type_list.append(list_of_l_names[index])
                            lambda_list.append(lambda_f)
                            tph_bgs_list.append(tph_bgs)
                            lambda_insertion_list.append(lambda_f * max(num_TTAAs, 1))

                        index = [lambda_bg, lambda_1k, lambda_5k, lambda_10k].index(
                            max([lambda_bg, lambda_1k, lambda_5k, lambda_10k])
                        )
                        lambdatype = list_of_l_names[index]

                    # number of insertions that are a user-defined distance from peak center
                    sig_flag = 0

    if record:
        peaks_frame = pd.DataFrame(
            columns=[
                "Chr",
                "Start",
                "End",
                "Center",
                "Experiment Insertions",
                "Fraction Experiment",
                "TPH Experiment",
                "Lambda Type",
                "Lambda",
                "Poisson pvalue",
            ]
        )

        peaks_frame["Lambda Reference Insertions"] = lambda_insertion_list
        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Insertions"] = num_exp_insertions_list
        peaks_frame["Fraction Experiment"] = frac_exp_list
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["Reference Insertions"] = num_bg_insertions_list
        peaks_frame["Fraction background"] = frac_bg_list
        peaks_frame["TPH background"] = tph_bg_list
        peaks_frame["TPH background subtracted"] = tph_bgs_list
        peaks_frame["Lambda Type"] = lambda_type_list
        peaks_frame["Lambda"] = lambda_list

    else:
        peaks_frame = pd.DataFrame(columns=["Chr", "Start", "End"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["Poisson pvalue"] = pvalue_list

    # peaks_frame = peaks_frame[peaks_frame["Poisson pvalue"] <= pvalue_cutoff]

    if record:
        return peaks_frame
    else:
        return peaks_frame[["Chr", "Start", "End"]]


def _callpeakscc_tools_bfnew2(
    expdata: pd.DataFrame,
    TTAAframe: pd.DataFrame,
    length: int,
    min_insertions: int = 3,
    extend: int = 200,
    window_size: int = 1000,
    step_size: int = 500,
    pseudocounts: float = 0.2,
    pvalue_cutoff: float = 0.01,
    lam_win_size: Optional[int] = None,
    record: bool = False,
    test_method: _PeakTestMethod = "poisson",
    multinumber=100000000,
) -> pd.DataFrame:

    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":
        from scipy.stats import binom_test

    # The chromosomes we need to consider
    chrm = list(expdata["Chr"].unique())

    # create lists to record
    chr_list = []
    start_list = []
    end_list = []
    pvalue_list = []
    sig_end = 0

    if record:
        center_list = []
        num_exp_insertions_list = []
        frac_exp_list = []
        tph_exp_list = []
        background_insertions = []
        expect_insertions = []

    total_experiment_insertions = len(expdata)

    for chrom in tqdm.tqdm(chrm):

        curTTAAframe = np.array(list(TTAAframe[TTAAframe["Chr"] == chrom]["Start"]))
        totalcurTTAA = len(curTTAAframe)
        if len(curTTAAframe) == 0:
            continue

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata["Chr"] == chrom]["Start"])
        curChromnp = np.array(curChrom)

        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()

        max_pos = curChrom[-1]
        sig_start = 0
        sig_end = 0
        sig_flag = 0

        totalcurChrom = len(curChromnp)

        startinsertions1 = 0
        startTTAA1 = 0

        startTTAA2 = 0

        if lam_win_size != None:

            startinsertionslam1 = 0
            startTTAAlam1 = 0

            startinsertionslam2 = 0
            startTTAAlam2 = 0

        if totalcurTTAA != 0:

            # caluclate the ratio for TTAA and background
            if lam_win_size == None:
                lambdacur = (
                    totalcurChrom / totalcurTTAA
                )  # expected ratio of insertions per TTAA

            for window_start in range(
                curChrom[0], int(max_pos + 2 * window_size), step_size
            ):

                if sig_end >= window_start:
                    continue

                num_exp_insertions, startinsertions1 = _findinsertionslen2(
                    curChromnp,
                    window_start,
                    window_start + window_size - 1,
                    length,
                    startinsertions1,
                    totalcurChrom,
                )

                if num_exp_insertions >= min_insertions:

                    num_TTAAs_window, startTTAA1 = _findinsertionslen2(
                        curTTAAframe,
                        window_start,
                        window_start + window_size - 1,
                        length,
                        startTTAA1,
                        totalcurTTAA,
                    )

                    # is this window significant?
                    if test_method == "poisson":

                        if lam_win_size == None:
                            pvalue = 1 - poisson.cdf(
                                (num_exp_insertions + pseudocounts),
                                lambdacur * max(num_TTAAs_window, 1) + pseudocounts,
                            )
                        else:
                            (
                                num_TTAA_insertions_lambda,
                                startTTAAlam1,
                            ) = _findinsertionslen2(
                                curTTAAframe,
                                window_start - int(lam_win_size / 2) + 1,
                                window_start + window_size + int(lam_win_size / 2) - 1,
                                length,
                                startTTAAlam1,
                                totalcurTTAA,
                            )

                            (
                                num_exp_insertions_lambda,
                                startinsertionslam1,
                            ) = _findinsertionslen2(
                                curChromnp,
                                window_start - int(lam_win_size / 2) + 1,
                                window_start + window_size + int(lam_win_size / 2) - 1,
                                length,
                                startinsertionslam1,
                                totalcurChrom,
                            )

                            pvalue = 1 - poisson.cdf(
                                (num_exp_insertions + pseudocounts),
                                float(
                                    num_exp_insertions_lambda
                                    / num_TTAA_insertions_lambda
                                )
                                * max(num_TTAAs_window, 1)
                                + pseudocounts,
                            )

                    elif test_method == "binomial":

                        if lam_win_size == None:
                            pvalue = binom_test(
                                int(num_exp_insertions + pseudocounts),
                                n=totalcurChrom,
                                p=((num_TTAAs_window + pseudocounts) / totalcurTTAA),
                                alternative="greater",
                            ).pvalue
                        else:
                            (
                                num_TTAA_insertions_lambda,
                                startTTAAlam1,
                            ) = _findinsertionslen2(
                                curTTAAframe,
                                window_start - int(lam_win_size / 2) + 1,
                                window_start + window_size + int(lam_win_size / 2) - 1,
                                length,
                                startTTAAlam1,
                                totalcurTTAA,
                            )
                            (
                                num_exp_insertions_lambda,
                                startinsertionslam1,
                            ) = _findinsertionslen2(
                                curChromnp,
                                window_start - int(lam_win_size / 2) + 1,
                                window_start + window_size + int(lam_win_size / 2) - 1,
                                length,
                                startinsertionslam1,
                                totalcurChrom,
                            )
                            pvalue = binom_test(
                                int(num_exp_insertions + pseudocounts),
                                n=num_exp_insertions_lambda,
                                p=(
                                    (num_TTAAs_window + pseudocounts)
                                    / num_TTAA_insertions_lambda
                                ),
                                alternative="greater",
                            ).pvalue

                else:
                    pvalue = 1

                if pvalue <= pvalue_cutoff:

                    # was last window significant?
                    if sig_flag:

                        # if so, extend end of windows
                        sig_end = window_start + window_size - 1

                    else:

                        # otherwise, define new start and end and set flag
                        sig_start = window_start
                        sig_end = window_start + window_size - 1
                        sig_flag = 1

                else:

                    # current window not significant.  Was last window significant?
                    if sig_flag:

                        # compute peak center and add to frame

                        overlap = _findinsertions(
                            curChromnp, sig_start, sig_end, length
                        )
                        peak_center = np.median(overlap)

                        # redefine the overlap
                        sig_start = overlap.min() - extend
                        sig_end = overlap.max() + length + extend
                        overlap = _findinsertions(
                            curChromnp, sig_start, sig_end, length
                        )
                        num_exp_insertions = len(overlap)

                        num_TTAAs_window, startTTAA2 = _findinsertionslen2(
                            curTTAAframe,
                            sig_start,
                            sig_end,
                            length,
                            startTTAA2,
                            totalcurTTAA,
                        )

                        # num_TTAAs_window= _findinsertionslen(curTTAAframe, sig_start, sig_end, length)
                        #                                                  startTTAA2, totalcurTTAA)

                        # compute pvalue and record it
                        if test_method == "poisson":

                            if lam_win_size == None:
                                pvalue_list.append(
                                    1
                                    - poisson.cdf(
                                        (num_exp_insertions + pseudocounts),
                                        lambdacur * max(num_TTAAs_window, 1)
                                        + pseudocounts,
                                    )
                                )
                            else:
                                (
                                    num_exp_insertions_lam_win_size,
                                    startinsertionslam2,
                                ) = _findinsertionslen2(
                                    curChromnp,
                                    peak_center - (lam_win_size / 2 - 1),
                                    peak_center + (lam_win_size / 2),
                                    length,
                                    startinsertionslam2,
                                    totalcurChrom,
                                )

                                (
                                    num_TTAAs_lam_win_size,
                                    startTTAAlam2,
                                ) = _findinsertionslen2(
                                    curTTAAframe,
                                    peak_center - (lam_win_size / 2 - 1),
                                    peak_center + (lam_win_size / 2),
                                    length,
                                    startTTAAlam2,
                                    totalcurTTAA,
                                )

                                lambda_lam_win_size = float(
                                    num_exp_insertions_lam_win_size
                                ) / (max(num_TTAAs_lam_win_size, 1))
                                pvalue_list.append(
                                    1
                                    - poisson.cdf(
                                        (num_exp_insertions + pseudocounts),
                                        lambda_lam_win_size * max(num_TTAAs_window, 1)
                                        + pseudocounts,
                                    )
                                )

                        elif test_method == "binomial":

                            if lam_win_size == None:
                                pvalue_list.append(
                                    binom_test(
                                        int(num_exp_insertions + pseudocounts),
                                        n=totalcurChrom,
                                        p=(
                                            (num_TTAAs_window + pseudocounts)
                                            / totalcurTTAA
                                        ),
                                        alternative="greater",
                                    ).pvalue
                                )
                            else:
                                (
                                    num_exp_insertions_lam_win_size,
                                    startinsertionslam2,
                                ) = _findinsertionslen2(
                                    curChromnp,
                                    peak_center - (lam_win_size / 2 - 1),
                                    peak_center + (lam_win_size / 2),
                                    length,
                                    startinsertionslam2,
                                    totalcurChrom,
                                )

                                (
                                    num_TTAAs_lam_win_size,
                                    startTTAAlam2,
                                ) = _findinsertionslen2(
                                    curTTAAframe,
                                    peak_center - (lam_win_size / 2 - 1),
                                    peak_center + (lam_win_size / 2),
                                    length,
                                    startTTAAlam2,
                                    totalcurTTAA,
                                )
                                pvalue_list.append(
                                    binom_test(
                                        int(num_exp_insertions + pseudocounts),
                                        n=num_exp_insertions_lam_win_size,
                                        p=(
                                            (num_TTAAs_window + pseudocounts)
                                            / num_TTAAs_lam_win_size
                                        ),
                                        alternative="greater",
                                    ).pvalue
                                )

                        chr_list.append(chrom)  # add chr to frame
                        start_list.append(max(sig_start, 0))  # add peak start to frame
                        end_list.append(sig_end)  # add peak end to frame

                        if record:

                            center_list.append(peak_center)  # add peak center to frame
                            num_exp_insertions_list.append(num_exp_insertions)

                            # add fraction of Experiment Insertions in peak to frame
                            frac_exp_list.append(
                                float(num_exp_insertions) / total_experiment_insertions
                            )
                            tph_exp_list.append(
                                float(num_exp_insertions)
                                * multinumber
                                / total_experiment_insertions
                            )

                            background_insertions.append(num_TTAAs_window)

                            if lam_win_size == None:
                                expect_insertions.append(
                                    lambdacur * max(num_TTAAs_window, 1) + pseudocounts
                                )
                            else:
                                expect_insertions.append(
                                    float(num_exp_insertions_lam_win_size)
                                    / (max(num_TTAAs_lam_win_size, 1))
                                    * max(num_TTAAs_window, 1)
                                    + pseudocounts
                                )

                        sig_flag = 0

    if record:
        peaks_frame = pd.DataFrame(
            columns=[
                "Chr",
                "Start",
                "End",
                "Center",
                "pvalue",
                "Experiment Insertions",
                "Reference Insertions",
                "Fraction Experiment",
                "TPH Experiment",
            ]
        )

        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Insertions"] = num_exp_insertions_list
        peaks_frame["Fraction Experiment"] = frac_exp_list
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["Reference Insertions"] = background_insertions
        peaks_frame["Expect insertions"] = expect_insertions

    else:
        peaks_frame = pd.DataFrame(columns=["Chr", "Start", "End", "pvalue"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["pvalue"] = pvalue_list
    peaks_frame = peaks_frame[peaks_frame["pvalue"] <= pvalue_cutoff]

    if record:
        return peaks_frame
    else:
        return peaks_frame[["Chr", "Start", "End"]]


def _callpeakscc_toolsnew2(
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
    test_method: _PeakTestMethod = "poisson",
    min_insertions: int = 3,
    record: bool = False,
) -> pd.DataFrame:

    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":
        from scipy.stats import binom_test

    multinumber = 100000000

    # The chromosomes we need to consider
    chrm = list(expdata["Chr"].unique())

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
        num_exp_insertions_list = []
        num_bg_insertions_list = []
        num_TTAA_insertions_list = []
        frac_exp_list = []
        tph_exp_list = []
        frac_bg_list = []
        tph_bg_list = []
        tph_bgs_list = []

    # record total number of insertions
    total_experiment_insertions = len(expdata)
    total_background_insertions = len(background)

    # going from the first Chromosome to the last
    for chrom in tqdm.tqdm(chrm):

        curbackgroundframe = np.array(
            list(background[background["Chr"] == chrom]["Start"])
        )
        totalcurbackground = len(curbackgroundframe)
        if totalcurbackground == 0:
            continue

        curTTAAframe = np.array(list(TTAAframe[TTAAframe["Chr"] == chrom]["Start"]))
        totalcurTTAA = len(curTTAAframe)
        if totalcurTTAA == 0:
            continue

        # find out the insertion data of our current chromosome
        curChrom = list(expdata[expdata["Chr"] == chrom]["Start"])
        curChromnp = np.array(curChrom)

        # sort it so that we could accelarate the searching afterwards
        curChromnp.sort()
        curbackgroundframe.sort()

        # initial the parameters
        max_pos = curChrom[-1] + length + 1
        sig_start = 0
        sig_end = 0
        sig_flag = 0

        # calculate the total number of insertions
        totalcurChrom = len(curChromnp)

        startinsertion1 = 0
        startTTAA1 = 0
        startbg1 = 0

        startinsertion2 = 0
        startTTAA2 = 0
        startbg2 = 0

        if lam_win_size == None:

            # caluclate the ratio for TTAA and background
            lambdacurTTAA = float(
                totalcurChrom / totalcurTTAA
            )  # expected ratio of insertions per TTAA

        else:

            startTTAAlam1 = 0
            startbglam1 = 0
            startinsertionlam = 0

            startTTAAlam2 = 0
            startbglam2 = 0
            startinsertionlam2 = 0

        for window_start in range(
            curChromnp[0], int(max_pos + 2 * window_size), int(step_size)
        ):

            if sig_end >= window_start:
                continue

            num_exp_insertions, startinsertion1 = _findinsertionslen2(
                curChromnp,
                window_start,
                window_start + window_size - 1,
                length,
                startinsertion1,
                totalcurChrom,
            )

            if num_exp_insertions >= min_insertions:

                # find out the number of insertions in the current window for backgound
                num_bg_insertions, startbg1 = _findinsertionslen2(
                    curbackgroundframe,
                    window_start,
                    window_start + window_size - 1,
                    length,
                    startbg1,
                    totalcurbackground,
                )

                if num_bg_insertions > 0:

                    if lam_win_size == None:

                        if test_method == "poisson":
                            pvaluebg = _compute_cumulative_poisson(
                                num_exp_insertions,
                                num_bg_insertions,
                                totalcurChrom,
                                totalcurbackground,
                                pseudocounts,
                            )
                        elif test_method == "binomial":
                            pvaluebg = binom_test(
                                int(num_exp_insertions + pseudocounts),
                                n=totalcurChrom,
                                p=(
                                    (num_bg_insertions + pseudocounts)
                                    / totalcurbackground
                                ),
                                alternative="greater",
                            ).pvalue
                    else:

                        num_exp_insertions_lam, startinsertionlam = _findinsertionslen2(
                            curChromnp,
                            window_start - int(lam_win_size / 2) + 1,
                            window_start + window_size + int(lam_win_size / 2) - 1,
                            length,
                            startinsertionlam,
                            totalcurChrom,
                        )

                        num_exp_bg_lam, startbglam1 = _findinsertionslen2(
                            curbackgroundframe,
                            window_start - int(lam_win_size / 2) + 1,
                            window_start + window_size + int(lam_win_size / 2) - 1,
                            length,
                            startbglam1,
                            totalcurbackground,
                        )

                        if test_method == "poisson":
                            pvaluebg = _compute_cumulative_poisson(
                                num_exp_insertions,
                                num_bg_insertions,
                                num_exp_insertions_lam,
                                num_exp_bg_lam,
                                pseudocounts,
                            )
                        elif test_method == "binomial":
                            pvaluebg = binom_test(
                                int(num_exp_insertions + pseudocounts),
                                n=num_exp_insertions_lam,
                                p=((num_bg_insertions + pseudocounts) / num_exp_bg_lam),
                                alternative="greater",
                            ).pvalue

                else:
                    if lam_win_size != None:
                        num_exp_insertions_lam, startinsertionlam = _findinsertionslen2(
                            curChromnp,
                            window_start - int(lam_win_size / 2) + 1,
                            window_start + window_size + int(lam_win_size / 2) - 1,
                            length,
                            startinsertionlam,
                            totalcurChrom,
                        )

                        num_exp_bg_lam, startbglam1 = _findinsertionslen2(
                            curbackgroundframe,
                            window_start - int(lam_win_size / 2) + 1,
                            window_start + window_size + int(lam_win_size / 2) - 1,
                            length,
                            startbglam1,
                            totalcurbackground,
                        )

                    pvaluebg = 0

                # if it passes, then look at the TTAA:
                if pvaluebg <= pvalue_cutoff_background:

                    num_TTAA_insertions, startTTAA1 = _findinsertionslen2(
                        curTTAAframe,
                        window_start,
                        window_start + window_size - 1,
                        length,
                        startTTAA1,
                        totalcurTTAA,
                    )

                    if lam_win_size == None:

                        if test_method == "poisson":
                            pvalueTTAA = 1 - poisson.cdf(
                                (num_exp_insertions + pseudocounts),
                                lambdacurTTAA * num_TTAA_insertions + pseudocounts,
                            )
                        elif test_method == "binomial":
                            pvalueTTAA = binom_test(
                                int(num_exp_insertions + pseudocounts),
                                n=totalcurChrom,
                                p=((num_TTAA_insertions + pseudocounts) / totalcurTTAA),
                                alternative="greater",
                            ).pvalue
                    else:

                        num_TTAA_insertions_lam, startTTAAlam1 = _findinsertionslen2(
                            curTTAAframe,
                            window_start - int(lam_win_size / 2) + 1,
                            window_start + window_size + int(lam_win_size / 2) - 1,
                            length,
                            startTTAAlam1,
                            totalcurTTAA,
                        )
                        if test_method == "poisson":
                            pvalueTTAA = 1 - poisson.cdf(
                                (num_exp_insertions + pseudocounts),
                                (num_exp_insertions_lam / num_TTAA_insertions_lam)
                                * num_TTAA_insertions
                                + pseudocounts,
                            )
                        elif test_method == "binomial":
                            pvalueTTAA = binom_test(
                                int(num_exp_insertions + pseudocounts),
                                n=num_exp_insertions_lam,
                                p=(
                                    (num_TTAA_insertions + pseudocounts)
                                    / num_TTAA_insertions_lam
                                ),
                                alternative="greater",
                            ).pvalue

                else:
                    pvaluebg = 1
                    pvalueTTAA = 1

            else:
                pvaluebg = 1
                pvalueTTAA = 1

            # is this window significant?
            if (
                pvaluebg <= pvalue_cutoff_background
                and pvalueTTAA <= pvalue_cutoff_TTAA
            ):
                # was last window significant?
                if sig_flag:
                    # if so, extend end of windows
                    sig_end = window_start + window_size - 1
                else:
                    # otherwise, define new start and end and set flag
                    sig_start = window_start
                    sig_end = window_start + window_size - 1
                    sig_flag = 1

            else:
                # current window not significant.  Was last window significant?
                if sig_flag:

                    # Let's first give a initial view of our peak
                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    peak_center = np.median(overlap)

                    # redefine the overlap
                    sig_start = overlap.min() - extend
                    sig_end = overlap.max() + 3 + extend

                    overlap = _findinsertions(curChromnp, sig_start, sig_end, length)
                    num_exp_insertions = len(overlap)

                    # add number of background insertions in peak to frame
                    num_TTAA_insertions, startTTAA2 = _findinsertionslen2(
                        curTTAAframe,
                        sig_start,
                        sig_end,
                        length,
                        startTTAA2,
                        totalcurTTAA,
                    )
                    num_bg_insertions, startbg2 = _findinsertionslen2(
                        curbackgroundframe,
                        sig_start,
                        sig_end,
                        length,
                        startbg2,
                        totalcurbackground,
                    )

                    chr_list.append(chrom)  # add chr to frame
                    start_list.append(max(sig_start, 0))  # add peak start to frame
                    end_list.append(sig_end)

                    if record:
                        # add peak end to frame
                        center_list.append(peak_center)  # add peak center to frame
                        num_TTAA_insertions_list.append(num_TTAA_insertions)
                        num_exp_insertions_list.append(
                            num_exp_insertions
                        )  # add fraction of Experiment Insertions in peak to frame
                        frac_exp_list.append(
                            float(num_exp_insertions) / total_experiment_insertions
                        )
                        tph_exp_list.append(
                            float(num_exp_insertions)
                            * multinumber
                            / total_experiment_insertions
                        )
                        num_bg_insertions_list.append(num_bg_insertions)
                        frac_bg_list.append(
                            float(num_bg_insertions) / total_background_insertions
                        )
                        tph_bg_list.append(
                            float(num_bg_insertions)
                            * multinumber
                            / total_background_insertions
                        )
                        tph_bgs = (
                            float(num_exp_insertions)
                            * multinumber
                            / total_experiment_insertions
                            - float(num_bg_insertions)
                            * multinumber
                            / total_background_insertions
                        )
                        tph_bgs_list.append(tph_bgs)

                    # caluclate the final P value

                    if lam_win_size == None:

                        if test_method == "poisson":

                            pvalue_list_TTAA.append(
                                1
                                - poisson.cdf(
                                    (num_exp_insertions + pseudocounts),
                                    lambdacurTTAA * num_TTAA_insertions + pseudocounts,
                                )
                            )
                            pvalue_list_background.append(
                                _compute_cumulative_poisson(
                                    num_exp_insertions,
                                    num_bg_insertions,
                                    totalcurChrom,
                                    totalcurbackground,
                                    pseudocounts,
                                )
                            )

                        elif test_method == "binomial":

                            pvalue_list_TTAA.append(
                                binom_test(
                                    int(num_exp_insertions + pseudocounts),
                                    n=totalcurChrom,
                                    p=(
                                        (num_TTAA_insertions + pseudocounts)
                                        / totalcurTTAA
                                    ),
                                    alternative="greater",
                                ).pvalue
                            )
                            pvalue_list_background.append(
                                binom_test(
                                    int(num_exp_insertions + pseudocounts),
                                    n=totalcurChrom,
                                    p=(
                                        (num_bg_insertions + pseudocounts)
                                        / totalcurbackground
                                    ),
                                    alternative="greater",
                                ).pvalue
                            )
                    else:

                        (
                            num_exp_insertions_lam,
                            startinsertionlam2,
                        ) = _findinsertionslen2(
                            curChromnp,
                            sig_start - int(lam_win_size / 2) + 1,
                            sig_end + int(lam_win_size / 2) - 1,
                            length,
                            startinsertionlam2,
                            totalcurChrom,
                        )

                        num_exp_bg_lam, startbglam2 = _findinsertionslen2(
                            curbackgroundframe,
                            sig_start - int(lam_win_size / 2) + 1,
                            sig_end + int(lam_win_size / 2) - 1,
                            length,
                            startbglam2,
                            totalcurbackground,
                        )

                        num_exp_TTAA_lam, startTTAAlam2 = _findinsertionslen2(
                            curTTAAframe,
                            sig_start - int(lam_win_size / 2) + 1,
                            sig_end + int(lam_win_size / 2) - 1,
                            length,
                            startTTAAlam2,
                            totalcurTTAA,
                        )

                        if test_method == "poisson":

                            pvalue_list_TTAA.append(
                                1
                                - poisson.cdf(
                                    (num_exp_insertions + pseudocounts),
                                    (num_exp_insertions_lam / num_exp_TTAA_lam)
                                    * num_TTAA_insertions
                                    + pseudocounts,
                                )
                            )
                            pvalue_list_background.append(
                                _compute_cumulative_poisson(
                                    num_exp_insertions,
                                    num_bg_insertions,
                                    num_exp_insertions_lam,
                                    num_exp_bg_lam,
                                    pseudocounts,
                                )
                            )

                        elif test_method == "binomial":

                            pvalue_list_TTAA.append(
                                binom_test(
                                    int(num_exp_insertions + pseudocounts),
                                    n=num_exp_insertions_lam,
                                    p=(
                                        (num_TTAA_insertions + pseudocounts)
                                        / num_exp_TTAA_lam
                                    ),
                                    alternative="greater",
                                ).pvalue
                            )
                            if num_exp_bg_lam == 0:
                                pvalue_list_background.append(0)
                            else:
                                pvalue_list_background.append(
                                    binom_test(
                                        int(num_exp_insertions + pseudocounts),
                                        n=num_exp_insertions_lam,
                                        p=(
                                            (num_bg_insertions + pseudocounts)
                                            / num_exp_bg_lam
                                        ),
                                        alternative="greater",
                                    ).pvalue
                                )

                    # number of insertions that are a user-defined distance from peak center
                    sig_flag = 0

    if record:
        peaks_frame = pd.DataFrame(
            columns=[
                "Chr",
                "Start",
                "End",
                "Center",
                "Experiment Insertions",
                "Background insertions",
                "Reference Insertions",
                "pvalue Reference",
                "pvalue Background",
            ]
        )

        peaks_frame["Center"] = center_list
        peaks_frame["Experiment Insertions"] = num_exp_insertions_list
        peaks_frame["Background insertions"] = num_bg_insertions_list
        peaks_frame["Reference Insertions"] = num_TTAA_insertions_list

        peaks_frame["Fraction Experiment"] = frac_exp_list
        peaks_frame["TPH Experiment"] = tph_exp_list
        peaks_frame["Fraction background"] = frac_bg_list
        peaks_frame["TPH background"] = tph_bg_list
        peaks_frame["TPH background subtracted"] = tph_bgs_list

    else:
        peaks_frame = pd.DataFrame(columns=["Chr", "Start", "End"])

    peaks_frame["Chr"] = chr_list
    peaks_frame["Start"] = start_list
    peaks_frame["End"] = end_list
    peaks_frame["pvalue Reference"] = pvalue_list_TTAA
    peaks_frame["pvalue Background"] = pvalue_list_background

    peaks_frame = peaks_frame[peaks_frame["pvalue Reference"] <= pvalue_cutoff_TTAA]
    peaks_frame = peaks_frame[
        peaks_frame["pvalue Background"] <= pvalue_cutoff_background
    ]

    if record:
        return peaks_frame
    else:
        return peaks_frame[["Chr", "Start", "End"]]


def _checkint(number, name):

    try:
        number = int(number)
    except:
        print("Please enter a valid positive number or 0 for" + name)
    if number < 0:
        raise ValueError("Please enter a valid positive number or 0 for" + name)

    return number


def _checkpvalue(number, name):

    try:
        number = float(number)
    except:
        print("Please enter a valid number (0,1) for " + name)
    if number < 0 or number > 1:
        raise ValueError("Please enter a valid number (0,1) for " + name)

    return number


def _check_test_method(method):
    if method != "poisson" and method != "binomial":
        raise ValueError(
            "Not valid a valid CCcaller method. Please input poisson or binomial."
        )


def call_peaks(
    expdata: pd.DataFrame,
    background: Optional[pd.DataFrame] = None,
    method: _Peakcalling_Method = "CCcaller",
    reference: _reference = "hg38",
    pvalue_cutoff: float = 0.0001,
    pvalue_cutoffbg: float = 0.0001,
    pvalue_cutoffTTAA: float = 0.00001,
    pvalue_adj_cutoff: Optional[float] = None,
    min_insertions: int = 5,
    minlen: int = 0,
    extend: int = 200,
    maxbetween: int = 2000,
    minnum: int = 0,
    test_method: _PeakTestMethod = "poisson",
    window_size: int = 1500,
    lam_win_size: Optional[int] = 100000,
    step_size: int = 500,
    pseudocounts: float = 0.2,
    min_length: int = None,
    max_length: int = None,
    record: bool = True,
    save: Optional[str] = None,
) -> pd.DataFrame:

    """\
    Call peaks from qbed data.

    :param expdata:
        pd.DataFrame with the first three columns as chromosome, start and end.
    :param background: Default is `None` for backgound free situation.
        pd.DataFrame with the first three columns as chromosome, start and end.
    :param method:
        `'CCcaller'` is a method considering the maxdistance between insertions in the data,
        `'cc_tools'` uses the idea adapted from :cite:`zhang2008model` and
        `here <https://hbctraining.github.io/Intro-to-ChIPseq/lessons/05_peak_calling_macs.html>`__.
        `'Blockify'` uses the method from :cite:`moudgil2020self` and `here <https://blockify.readthedocs.io/en/latest/>`__.
    :param reference:
        We currently have `'hg38'` for human data, `'mm10'` for mouse data and `'sacCer3'` for yeast data.
    :param pvalue_cutoff:
        The P-value cutoff for a backgound free situation.
    :param pvalue_cutoffbg:
        The P-value cutoff for backgound data when backgound exists.
    :param pvalue_cutoffTTAA:
        The P-value cutoff for reference data when backgound exists.
        Note that pvalue_cutoffTTAA is recommended to be lower than pvalue_cutoffbg.
    :param pvalue_adj_cutoff:
        The cutoff for the adjusted pvalue.
    :param min_insertions:
        The number of minimal insertions for each peak.
    :param minlen:
        Valid only for method = `'CCcaller'`. The minimal length for a peak without extend.
    :param extend:
        Valid for method = `'CCcaller'` and `'cc_tools'`. The length (bp) that peaks extend for both sides.
    :param maxbetween:
        Valid only for method = `'CCcaller'`. The maximum length of nearby position within one peak.
    :param minnum:
        Valid only for method = `'CCcaller'`. The minmum number of insertions for the nearby position.
    :param test_method:
        The method for making hypothesis.
    :param window_size:
        Valid only for method = `'cc_tools'`. The length of window looking for.
    :param lam_win_size:
        Valid for  method = `'CCcaller'` and `'cc_tools'`. The length of peak area considered when performing a CCcaller.
    :param step_size:
        Valid only for `'cc_tools'`. The length of each step.
    :param pseudocounts:
        Number for pseudocounts added for the pyhothesis.
    :param min_length:
        minimum length of peak, valid for Blockify.
    :param max_length:
        maximum length of peak, valid for Blockify.
    :param record:
        Controls if information is recorded.
        If `False`, the output would only have three columns: Chromosome, Start, End.
    :param save:
        The file name for the file we saved.


    :Returns:
        | **Chr** - The chromosome of the peak.
        | **Start** - The start point of the peak.
        | **End** - The end point of the peak.
        | **Experiment Insertions** - The total number of insertions within a peak in the experiment data.
        | **Reference Insertions** - The total number of insertions of within a peak in the reference data.
        | **Background insertions** - The total number of insertions within a peak in the experiment data.
        | **Expected Insertions** - The total number of expected insertions under null hypothesis from the reference data (in a background free situation).
        | **Expected Insertions background** - The total number of expected insertions under null hypothesis from the background data (in a background situation).
        | **Expected Insertions Reference** - The total number of expected insertions under null hypothesis from the reference data (in a background situation).
        | **pvalue** - The pvalue we calculate from null hypothesis (in a background free situation or method = `'Blockify'`).
        | **pvalue Reference** - The total number of insertions of within a peak in the reference data (in a background situation).
        | **pvalue Background** - The total number of insertions of within a peak in the reference data (in a background situation).
        | **Fraction Experiment** - The fraction of insertions  in the experiment data.
        | **TPH Experiment** - Transpositions per hundred million insertions in the experiment data for mammalian and
                               transpositions per hundred million insertions in the experiment data for sacCer3.
        | **Fraction Background** - The fraction of insertions in the background data.
        | **TPH Background** - Transpositions per hundred million insertions in the background data for mammalian and
                               transpositions per hundred million insertions in the background data for sacCer3.
        | **TPH Background subtracted** - The difference between TPH Experiment and TPH Background.


    :Examples:
    >>> import pycallingcards as cc
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.call_peaks(qbed_data, method = "CCcaller", reference = "mm10",  maxbetween = 2000,pvalue_cutoff = 0.01, pseudocounts = 1, record = True)

    """

    if type(expdata) != pd.DataFrame:
        raise ValueError("Please input a pandas dataframe as the expression data.")

    if type(record) != bool:
        raise ValueError("Please enter a True/ False for record")

    if save == True:
        save = "peak.bed"

    if type(background) == pd.DataFrame:

        if pvalue_adj_cutoff == None:
            pvalue_adj_cutoff = pvalue_cutoffTTAA

        length = 3

        if method == "cc_tools":

            print(
                "For the cc_tools method with background, [expdata, background, reference, pvalue_cutoffbg, pvalue_cutoffTTAA, lam_win_size, window_size, step_size, extend, pseudocounts, test_method, min_insertions, record] would be utilized."
            )

            if reference == "hg38":
                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
            elif reference == "mm10":
                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
            else:
                raise ValueError("Not valid reference.")

            _checkpvalue(pvalue_cutoffbg, "pvalue_cutoffbg")
            _checkpvalue(pvalue_cutoffTTAA, "pvalue_cutoffTTAA")

            window_size = _checkint(window_size, "window_size")
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size, "lam_win_size")
            extend = _checkint(extend, "extend")
            step_size = _checkint(step_size, "step_size")

            _check_test_method(test_method)
            min_insertions = _checkint(min_insertions, "min_insertions")
            min_insertions = max(min_insertions, 1)

            return_data = _callpeakscc_toolsnew2(
                expdata,
                background,
                TTAAframe,
                length,
                extend=extend,
                lam_win_size=lam_win_size,
                pvalue_cutoff_background=pvalue_cutoffbg,
                pvalue_cutoff_TTAA=pvalue_cutoffTTAA,
                window_size=window_size,
                step_size=step_size,
                pseudocounts=pseudocounts,
                test_method=test_method,
                min_insertions=min_insertions,
                record=record,
            ).reset_index(drop=True)

            return_data = _fdrcorrection(
                return_data,
                pvalue_adj_cutoff,
                reference,
                pvalue_before="pvalue Reference",
                pvalue_after="pvalue_adj Reference",
            )

            if save == None or save == False:

                return return_data
            else:

                return_data.to_csv(save, sep="\t", header=None, index=None)

                return return_data

        elif method == "CCcaller":

            print(
                "For the CCcaller method with background, [expdata, background, reference, pvalue_cutoffbg, pvalue_cutoffTTAA, lam_win_size, pseudocounts, minlen, extend, maxbetween, test_method, min_insertions, record] would be utilized."
            )

            if reference == "hg38":
                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
            elif reference == "mm10":
                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
            else:
                raise ValueError("Not valid reference.")

            _checkpvalue(pvalue_cutoffbg, "pvalue_cutoffbg")
            _checkpvalue(pvalue_cutoffTTAA, "pvalue_cutoffTTAA")

            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size, "lam_win_size")
            extend = _checkint(extend, "extend")
            _checkint(pseudocounts, "pseudocounts")
            _check_test_method(test_method)

            minlen = _checkint(minlen, "minlen")
            min_insertions = _checkint(min_insertions, "min_insertions")
            min_insertions = max(min_insertions, 1)
            maxbetween = _checkint(maxbetween, "maxbetween")

            _check_test_method(test_method)
            min_insertions = _checkint(min_insertions, "min_insertions")
            min_insertions = max(min_insertions, 1)

            return_data = _CCcaller2(
                expdata,
                background,
                TTAAframe,
                length,
                pvalue_cutoffbg=pvalue_cutoffbg,
                pvalue_cutoffTTAA=pvalue_cutoffTTAA,
                mininser=min_insertions,
                minlen=minlen,
                extend=extend,
                maxbetween=maxbetween,
                lam_win_size=lam_win_size,
                pseudocounts=pseudocounts,
                test_method=test_method,
                record=record,
                minnum=minnum,
            )

            return_data = _fdrcorrection(
                return_data,
                pvalue_adj_cutoff,
                reference,
                pvalue_before="pvalue Reference",
                pvalue_after="pvalue_adj Reference",
            )

            if save == None or save == False:

                return return_data
            else:

                return_data.to_csv(save, sep="\t", header=None, index=None)

                return return_data

        elif method == "Blockify":

            print(
                "For the Blockify method with background, [expdata, background, pvalue_cutoff, pseudocounts, test_method, min_length, min_length, record] would be utilized."
            )

            if type(record) != bool:
                raise ValueError("Please enter a True/ False for record")

            _check_test_method(test_method)
            _checkpvalue(pvalue_cutoff, "pvalue_cutoff")
            _checkint(pseudocounts, "pseudocounts")

            return_data = _Blockify(
                expdata,
                background,
                length,
                pvalue_cutoff=pvalue_cutoff,
                pseudocounts=pseudocounts,
                test_method=test_method,
                record=record,
            )

            if max_length != None:
                return_data = return_data[
                    return_data["End"] - return_data["Start"] <= max_length
                ]

            if min_length != None:
                return_data = return_data[
                    return_data["End"] - return_data["Start"] >= min_length
                ]

            return_data = _fdrcorrection(return_data, pvalue_adj_cutoff, reference)

            if save == None or save == False:

                return return_data

            else:

                return_data.to_csv(save, sep="\t", header=None, index=None)

                return return_data

        if method == "cc_tools_old":

            print(
                "For the cc_tools method with background, [expdata, background, reference, pvalue, lam_win_size, window_size, step_size,pseudocounts,  record] would be utilized."
            )

            if reference == "hg38":
                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
            elif reference == "mm10":
                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
            else:
                raise ValueError("Not valid reference.")

            _checkpvalue(pvalue_cutoff, "pvalue_cutoff")

            window_size = _checkint(window_size, "window_size")
            lam_win_size = _checkint(lam_win_size, "lam_win_size")
            step_size = _checkint(step_size, "step_size")

            return_data = _callpeakscc_tools(
                expdata,
                background,
                TTAAframe,
                length,
                window_size=window_size,
                lam_win_size=lam_win_size,
                step_size=step_size,
                pseudocounts=pseudocounts,
                pvalue_cutoff=pvalue_cutoff,
                record=record,
            ).reset_index(drop=True)

            return_data = _fdrcorrection(
                return_data,
                pvalue_adj_cutoff,
                reference,
                pvalue_before="pvalue Reference",
                pvalue_after="pvalue_adj Reference",
            )

            if save == None or save == False:

                return return_data

            else:

                return_data.to_csv(save, sep="\t", header=None, index=None)

                return return_data

        else:

            raise ValueError("Not valid Method.")

    if background == None:

        if pvalue_adj_cutoff == None:
            pvalue_adj_cutoff = pvalue_cutoff

        if method == "cc_tools":

            print(
                "For the cc_tools method without background, [expdata, reference, pvalue_cutoff, lam_win_size, window_size, step_size, extend, pseudocounts, test_method, min_insertions, record] would be utilized."
            )

            if reference == "hg38":

                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
                length = 3
                multinumber = 100000000

            elif reference == "mm10":

                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
                length = 3
                multinumber = 100000000

            elif reference == "sacCer3":

                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/yeast_Background.ccf",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End", "Reads"],
                )
                length = 0
                multinumber = 100000000

            else:
                raise ValueError("Not valid reference.")

            _checkpvalue(pvalue_cutoff, "pvalue_cutoff")

            window_size = _checkint(window_size, "window_size")
            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size, "lam_win_size")
            extend = _checkint(extend, "extend")
            step_size = _checkint(step_size, "step_size")
            _checkint(pseudocounts, "pseudocounts")

            _check_test_method(test_method)
            min_insertions = _checkint(min_insertions, "min_insertions")
            min_insertions = max(min_insertions, 1)

            return_data = _callpeakscc_tools_bfnew2(
                expdata,
                TTAAframe,
                length,
                extend=extend,
                pvalue_cutoff=pvalue_cutoff,
                window_size=window_size,
                lam_win_size=lam_win_size,
                step_size=step_size,
                pseudocounts=pseudocounts,
                test_method=test_method,
                min_insertions=min_insertions,
                record=record,
                multinumber=multinumber,
            ).reset_index(drop=True)

            return_data = _fdrcorrection(
                return_data,
                pvalue_adj_cutoff,
                reference,
                pvalue_before="pvalue",
                pvalue_after="pvalue_adj",
            )

            if save == None or save == False:

                return return_data
            else:

                return_data.to_csv(save, sep="\t", header=None, index=None)

                return return_data

        elif method == "CCcaller":

            print(
                "For the CCcaller method without background, [expdata, reference, pvalue_cutoff, lam_win_size, pseudocounts, minlen, extend, maxbetween, test_method, min_insertions, record] would be utilized."
            )

            if reference == "hg38":

                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
                length = 3

            elif reference == "mm10":

                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
                length = 3

            elif reference == "sacCer3":

                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/yeast_Background.ccf",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End", "Reads"],
                )
                length = 0

            else:
                raise ValueError("Not valid reference.")

            _checkpvalue(pvalue_cutoff, "pvalue_cutoff")

            if lam_win_size != None:
                lam_win_size = _checkint(lam_win_size, "lam_win_size")
            extend = _checkint(extend, "extend")
            _checkint(pseudocounts, "pseudocounts")
            _check_test_method(test_method)

            minlen = _checkint(minlen, "minlen")
            min_insertions = _checkint(min_insertions, "min_insertions")
            min_insertions = max(min_insertions, 1)
            maxbetween = _checkint(maxbetween, "maxbetween")

            _check_test_method(test_method)
            min_insertions = _checkint(min_insertions, "min_insertions")
            min_insertions = max(min_insertions, 1)

            return_data = _CCcaller_bf2(
                expdata,
                TTAAframe,
                length,
                pvalue_cutoff=pvalue_cutoff,
                mininser=min_insertions,
                minlen=minlen,
                extend=extend,
                maxbetween=maxbetween,
                lam_win_size=lam_win_size,
                pseudocounts=pseudocounts,
                test_method=test_method,
                record=record,
                minnum=minnum,
            )

            return_data = _fdrcorrection(
                return_data,
                pvalue_adj_cutoff,
                reference,
                pvalue_before="pvalue",
                pvalue_after="pvalue_adj",
            )

            if save == None or save == False:

                return return_data

            else:

                return_data.to_csv(save, sep="\t", header=None, index=None)

                return return_data

        elif method == "Blockify":

            print(
                "For the Blockify method with background, [expdata, reference, pvalue_cutoff, pseudocounts, test_method,  record] would be utilized."
            )

            if reference == "hg38":

                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
                length = 3

            elif reference == "mm10":

                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End"],
                )
                length = 3

            elif reference == "sacCer3":

                TTAAframe = pd.read_csv(
                    "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/yeast_Background.ccf",
                    delimiter="\t",
                    header=None,
                    names=["Chr", "Start", "End", "Reads"],
                )
                length = 0

            else:
                raise ValueError("Not valid reference.")

            _checkint(pseudocounts, "pseudocounts")

            if type(record) != bool:
                raise ValueError("Please enter a True/ False for record")

            _check_test_method(test_method)
            _checkpvalue(pvalue_cutoff, "pvalue_cutoff")

            return_data = _Blockify(
                expdata,
                TTAAframe,
                length,
                pvalue_cutoff=pvalue_cutoff,
                pseudocounts=pseudocounts,
                test_method=test_method,
                record=record,
            )

            if max_length != None:
                return_data = return_data[
                    return_data["End"] - return_data["Start"] <= max_length
                ]

            if min_length != None:
                return_data = return_data[
                    return_data["End"] - return_data["Start"] >= min_length
                ]

            return_data = _fdrcorrection(
                return_data,
                pvalue_adj_cutoff,
                reference,
                pvalue_before="pvalue",
                pvalue_after="pvalue_adj",
            )

            if save == None or save == False:

                return return_data

            else:

                return_data.to_csv(save, sep="\t", header=None, index=None)

                return return_data

        else:

            raise ValueError("Not valid Method.")

    else:

        raise ValueError("Not a valid background.")


def down_sample(
    qbed: pd.DataFrame,
    number: int = 10000000,
    random_state: int = 1,
):

    """\
    Down sample insertion data.

    :param qbed:
        pd.DataFrame for qbed data.
    :param number:
        The target number for downsampling. It should be less than the total number of insertions.
    :param random_state:
        The random seed.


    :Examples:
    >>> import pycallingcards as cc
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.down_sample(qbed_data, number = 10000)

    """

    qbed_ram = qbed.sample(n=number, random_state=random_state)
    qbed_ram = qbed_ram.sort_values(by=["Chr", "Start"])

    return qbed_ram


def _closest(lst, K):
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return idx


def combine_peaks(
    peak_data: pd.DataFrame,
    index: int,
    expdata: Optional[pd.DataFrame] = None,
    background: Optional[pd.DataFrame] = None,
    method: _Peakcalling_Method = "CCcaller",
    reference: _reference = "hg38",
    test_method: _Peakcalling_Method = "poisson",
    lam_win_size: Optional[int] = 100000,
    pvalue_cutoff: Optional[float] = None,
    pvalue_cutoffbg: Optional[float] = None,
    pvalue_cutoffTTAA: Optional[float] = None,
    pseudocounts: float = 0.2,
    return_whole: bool = False,
):

    """\
    Combine two peaks.

    This function combine the one and the next peak peaks.

    :param peak_data:
        pd.DataFrame for peak data. Please input the original data from call_peaks function.
    :param index:
        The index for the first peak to combine. Will combine peak index and peak index+1.
    :param expdata:
        pd.DataFrame with the first three columns as chromosome, start and end.
    :param background:
        pd.DataFrame with the first three columns as chromosome, start and end.
    :param method:
        `'CCcaller'` is a method considering the maxdistance between insertions in the data,
        `'cc_tools'` uses the idea adapted from :cite:`zhang2008model` and
        `here <https://hbctraining.github.io/Intro-to-ChIPseq/lessons/05_peak_calling_macs.html>`__.
        `'Blockify'` uses the method from :cite:`moudgil2020self` and `here <https://blockify.readthedocs.io/en/latest/>`__.
    :param reference:
        We currently have `'hg38'` for human data, `'mm10'` for mouse data and `'sacCer3'` for yeast data.
    :param pvalue_cutoff:
        The P-value cutoff for a backgound free situation. If None, no filteration.
    :param pvalue_cutoffbg:
        The P-value cutoff for backgound data when backgound exists. If None, no filteration.
    :param pvalue_cutoffTTAA:
        The P-value cutoff for reference data when backgound exists.
        Note that pvalue_cutoffTTAA is recommended to be lower than pvalue_cutoffbg. If None, no filteration.
    :param pseudocounts:
        Number for pseudocounts added for the pyhothesis.
    :param return_whole:
        If False, return only the combined peak.
        If True, return the whole peak dataframe.

    :Examples:
    >>> import pycallingcards as cc
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.call_peaks(qbed_data, method = "CCcaller", reference = "mm10",  maxbetween = 2000,pvalue_cutoff = 0.01, pseudocounts = 1, record = True)
    >>> peak_data = cc.pp.combine_peaks(peak_data, 1, qbed_data, method = "CCcaller", reference = "mm10",  pvalue_cutoff = 0.01, pseudocounts = 1, return_whole = True)


    """

    peak_data_temp = peak_data.copy()

    if test_method == "binomial":
        from scipy.stats import binom_test

    chrom1 = peak_data_temp.iloc[index, 0]
    chrom2 = peak_data_temp.iloc[index, 0]

    if chrom1 != chrom2:
        raise Exception("Cannot combine peaks from different chromosomes.")

    start = peak_data_temp.iloc[index, 1]
    end = peak_data_temp.iloc[index + 1, 2]

    index_list = list(peak_data_temp.columns)
    totallen = len(index_list)

    if totallen == 3:

        if return_whole == False:
            return pd.DataFrame([[chrom1, start, end]], columns=index_list)
        else:
            peak_data_temp.iloc[index, 2] = end
            peak_data_temp = peak_data_temp.drop(index + 1)
            return peak_data_temp.reset_index(drop=True)

    if reference == "hg38":
        TTAA_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",
            sep="\t",
            header=None,
        )
        length = 4

    elif reference == "mm10":
        TTAA_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",
            sep="\t",
            header=None,
        )
        length = 4

    elif reference == "sacCer3":
        TTAA_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/yeast_Background.ccf",
            sep="\t",
            header=None,
        )
        length = 1

    if type(expdata) == pd.DataFrame:
        expdatacounts = len(
            expdata[
                (expdata["Chr"] == chrom1)
                & (expdata["Start"] >= start - length)
                & (expdata["Start"] <= end)
            ]
        )
    else:
        raise Exception("Please input expdata.")

    print(
        "The new pvalue_adj is not the true one. It searches for the cloest one among the data."
    )

    if method == "CCcaller":

        TTAAcounts = len(
            TTAA_data[
                (TTAA_data[0] == chrom1)
                & (TTAA_data[2] >= start - length)
                & (TTAA_data[2] <= end)
            ]
        )
        if lam_win_size == None:
            expdatacounts_lam = len(expdata[(expdata["Chr"] == chrom1)])
            TTAAcounts_lam = len(TTAA_data[(TTAA_data[0] == chrom1)])
        else:
            expdatacounts_lam = len(
                expdata[
                    (expdata["Chr"] == chrom1)
                    & (expdata["Start"] >= start - length - lam_win_size / 2 + 1)
                    & (expdata["Start"] <= end + lam_win_size / 2)
                ]
            )
            TTAAcounts_lam = len(
                TTAA_data[
                    (TTAA_data[0] == chrom1)
                    & (TTAA_data[2] >= start - length - lam_win_size / 2 + 1)
                    & (TTAA_data[2] <= end + lam_win_size / 2)
                ]
            )

        expinsertion_TTAA = expdatacounts_lam * (TTAAcounts / TTAAcounts_lam)

        if test_method == "poisson":
            pvalue = _compute_cumulative_poisson(
                expdatacounts,
                TTAAcounts,
                expdatacounts_lam,
                TTAAcounts_lam,
                pseudocounts,
            )
        elif test_method == "binomial":
            pvalue = binom_test(
                int(expdatacounts + pseudocounts),
                n=expdatacounts_lam,
                p=((TTAAcounts + pseudocounts) / TTAAcounts_lam),
                alternative="greater",
            ).pvalue

        if type(background) == pd.DataFrame:

            backgroundcounts = len(
                background[
                    (background["Chr"] == chrom1)
                    & (background["Start"] >= start - length)
                    & (background["Start"] <= end)
                ]
            )
            if lam_win_size == None:
                backgroundcounts_lam = len(background[(background["Chr"] == chrom1)])
            else:
                backgroundcounts_lam = len(
                    background[
                        (background["Chr"] == chrom1)
                        & (background["Start"] >= start - length - lam_win_size / 2 + 1)
                        & (background["Start"] <= end + lam_win_size / 2)
                    ]
                )

            expinsertion_bg = expdatacounts_lam * (
                backgroundcounts / backgroundcounts_lam
            )

            if test_method == "poisson":
                pvalue_bg = _compute_cumulative_poisson(
                    expdatacounts,
                    backgroundcounts,
                    expdatacounts_lam,
                    backgroundcounts_lam,
                    pseudocounts,
                )
            elif test_method == "binomial":
                pvalue_bg = binom_test(
                    int(expdatacounts + pseudocounts),
                    n=expdatacounts_lam,
                    p=((backgroundcounts + pseudocounts) / backgroundcounts_lam),
                    alternative="greater",
                ).pvalue

            pvalue_adj = peak_data["pvalue_adj Reference"][
                _closest(list(peak_data["pvalue Reference"]), pvalue)
            ]
            # pvalue_adj_bg = peak_data["pvalue_adj Background"][_closest(list(peak_data["pvalue Background"]), pvalue_bg)]

            if (
                (pvalue_cutoffTTAA == None) or (pvalue <= float(pvalue_cutoffTTAA or 0))
            ) and (
                (pvalue_cutoffbg == None) or (pvalue_bg <= float(pvalue_cutoffbg or 0))
            ):

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom1,
                                start,
                                end,
                                expdatacounts,
                                backgroundcounts,
                                TTAAcounts,
                                expinsertion_bg,
                                expinsertion_TTAA,
                                pvalue_bg,
                                pvalue,
                                pvalue_adj,
                            ]
                        ],
                        columns=index_list,
                    )
                else:
                    peak_data_temp.iloc[index, 2] = end
                    peak_data_temp.iloc[index, 3] = expdatacounts
                    peak_data_temp.iloc[index, 4] = backgroundcounts
                    peak_data_temp.iloc[index, 5] = TTAAcounts
                    peak_data_temp.iloc[index, 6] = expinsertion_bg
                    peak_data_temp.iloc[index, 7] = expinsertion_TTAA
                    peak_data_temp.iloc[index, 8] = pvalue_bg
                    peak_data_temp.iloc[index, 9] = pvalue
                    # peak_data_temp.iloc[index,10] = pvalue_adj_bg
                    peak_data_temp.iloc[index, 10] = pvalue_adj
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    if totallen == 11:
                        return peak_data_temp.reset_index(drop=True)
                    else:
                        for i in range(11, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp.reset_index(drop=True)

            else:

                if return_whole == False:
                    return None
                else:
                    peak_data_temp = peak_data_temp.drop(index)
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    return peak_data_temp.reset_index(drop=True)

        else:

            pvalue_adj = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue)
            ]

            if (pvalue_cutoff == None) or (pvalue <= float(pvalue_cutoff or 0)):

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom1,
                                start,
                                end,
                                expdatacounts,
                                TTAAcounts,
                                expinsertion_TTAA,
                                pvalue,
                                pvalue_adj,
                            ]
                        ],
                        columns=index_list,
                    )

                else:
                    peak_data_temp.iloc[index, 2] = end
                    peak_data_temp.iloc[index, 3] = expdatacounts
                    peak_data_temp.iloc[index, 4] = TTAAcounts
                    peak_data_temp.iloc[index, 5] = expinsertion_TTAA
                    peak_data_temp.iloc[index, 6] = pvalue
                    peak_data_temp.iloc[index, 7] = pvalue_adj
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    if totallen == 8:
                        return peak_data_temp.reset_index(drop=True)
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp.reset_index(drop=True)

            else:

                if return_whole == False:
                    return None

                else:
                    peak_data_temp = peak_data_temp.drop(index)
                    peak_data_temp = peak_data_temp.drop(index + 1)
                    return peak_data_temp.reset_index(drop=True)

    elif method == "cc_tools":

        multinumber = 100000000
        sumcount_expdata = len(expdata)
        TTAAcounts = len(
            TTAA_data[
                (TTAA_data[0] == chrom1)
                & (TTAA_data[2] >= start - length)
                & (TTAA_data[2] <= end)
            ]
        )

        if lam_win_size == None:
            expdatacounts_lam = len(expdata[(expdata["Chr"] == chrom1)])
            TTAAcounts_lam = len(TTAA_data[(TTAA_data[0] == chrom1)])
        else:
            expdatacounts_lam = len(
                expdata[
                    (expdata["Chr"] == chrom1)
                    & (expdata["Start"] >= start - length - lam_win_size / 2 + 1)
                    & (expdata["Start"] <= end + lam_win_size / 2)
                ]
            )
            TTAAcounts_lam = len(
                TTAA_data[
                    (TTAA_data[0] == chrom1)
                    & (TTAA_data[2] >= start - length - lam_win_size / 2 + 1)
                    & (TTAA_data[2] <= end + lam_win_size / 2)
                ]
            )

        expinsertion_TTAA = expdatacounts_lam * (TTAAcounts / TTAAcounts_lam)
        counts_median = np.median(
            list(
                expdata[
                    (expdata["Chr"] == chrom1)
                    & (expdata["Start"] >= start - length)
                    & (expdata["Start"] <= end)
                ].iloc[:, 1]
            )
        )
        TPH = float(expdatacounts) * multinumber / sumcount_expdata
        frac_exp = float(expdatacounts) / sumcount_expdata

        if test_method == "poisson":
            pvalue = _compute_cumulative_poisson(
                expdatacounts,
                TTAAcounts,
                expdatacounts_lam,
                TTAAcounts_lam,
                pseudocounts,
            )
        elif test_method == "binomial":
            pvalue = binom_test(
                int(expdatacounts + pseudocounts),
                n=expdatacounts_lam,
                p=((TTAAcounts + pseudocounts) / TTAAcounts_lam),
                alternative="greater",
            ).pvalue

        if type(background) == pd.DataFrame:

            backgroundcounts = len(
                background[
                    (background["Chr"] == chrom1)
                    & (background["Start"] >= start - length)
                    & (background["Start"] <= end)
                ]
            )
            sumcount_background = len(background)
            if lam_win_size == None:
                backgroundcounts_lam = len(background[(background["Chr"] == chrom1)])
            else:
                backgroundcounts_lam = len(
                    background[
                        (background["Chr"] == chrom1)
                        & (background["Start"] >= start - length - lam_win_size / 2 + 1)
                        & (background["Start"] <= end + lam_win_size / 2)
                    ]
                )

            expinsertion_bg = expdatacounts_lam * (
                backgroundcounts / backgroundcounts_lam
            )
            TPH_bg = float(backgroundcounts) * multinumber / sumcount_background
            frac_exp_bg = float(backgroundcounts) / sumcount_background

            if test_method == "poisson":
                pvalue_bg = _compute_cumulative_poisson(
                    expdatacounts,
                    backgroundcounts,
                    expdatacounts_lam,
                    backgroundcounts_lam,
                    pseudocounts,
                )
            elif test_method == "binomial":
                pvalue_bg = binom_test(
                    int(expdatacounts + pseudocounts),
                    n=expdatacounts_lam,
                    p=((backgroundcounts + pseudocounts) / backgroundcounts_lam),
                    alternative="greater",
                ).pvalue

            pvalue_adj = peak_data["pvalue_adj Reference"][
                _closest(list(peak_data["pvalue Reference"]), pvalue)
            ]
            #    pvalue_adj_bg = peak_data["pvalue_adj Background"][_closest(list(peak_data["pvalue Background"]), pvalue_bg)]

            if (
                (pvalue_cutoffTTAA == None) or (pvalue <= float(pvalue_cutoffTTAA or 0))
            ) and (
                (pvalue_cutoffbg == None) or (pvalue_bg <= float(pvalue_cutoffbg or 0))
            ):

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom1,
                                start,
                                end,
                                counts_median,
                                expdatacounts,
                                backgroundcounts,
                                TTAAcounts,
                                pvalue,
                                pvalue_bg,
                                frac_exp,
                                TPH,
                                frac_exp_bg,
                                TPH_bg,
                                frac_exp - frac_exp_bg,
                                pvalue_adj,
                            ]
                        ],
                        columns=index_list,
                    )
                else:
                    peak_data_temp.iloc[index, 2] = end
                    peak_data_temp.iloc[index, 3] = counts_median
                    peak_data_temp.iloc[index, 4] = expdatacounts
                    peak_data_temp.iloc[index, 5] = backgroundcounts
                    peak_data_temp.iloc[index, 6] = TTAAcounts
                    peak_data_temp.iloc[index, 7] = pvalue
                    peak_data_temp.iloc[index, 8] = pvalue_bg
                    peak_data_temp.iloc[index, 9] = frac_exp
                    peak_data_temp.iloc[index, 10] = TPH
                    peak_data_temp.iloc[index, 11] = frac_exp_bg
                    peak_data_temp.iloc[index, 12] = TPH_bg
                    peak_data_temp.iloc[index, 13] = frac_exp - frac_exp_bg
                    #  peak_data_temp.iloc[index,14] = pvalue_adj_bg
                    peak_data_temp.iloc[index, 14] = pvalue_adj
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    if totallen == 15:
                        return peak_data_temp.reset_index()
                    else:
                        for i in range(15, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp.reset_index()

            else:

                if return_whole == False:
                    return None

                else:
                    peak_data_temp = peak_data_temp.drop(index)
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    return peak_data_temp.reset_index()

        else:

            pvalue_adj = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue)
            ]

            if (pvalue_cutoff == None) or (pvalue <= float(pvalue_cutoff or 0)):

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom1,
                                start,
                                end,
                                counts_median,
                                pvalue,
                                expdatacounts,
                                TTAAcounts,
                                frac_exp,
                                TPH,
                                expinsertion_TTAA,
                                pvalue_adj,
                            ]
                        ],
                        columns=index_list,
                    )
                else:
                    peak_data_temp.iloc[index, 2] = end
                    peak_data_temp.iloc[index, 3] = counts_median
                    peak_data_temp.iloc[index, 4] = pvalue
                    peak_data_temp.iloc[index, 5] = expdatacounts
                    peak_data_temp.iloc[index, 6] = TTAAcounts
                    peak_data_temp.iloc[index, 7] = frac_exp
                    peak_data_temp.iloc[index, 8] = TPH
                    peak_data_temp.iloc[index, 9] = expinsertion_TTAA
                    peak_data_temp.iloc[index, 10] = pvalue_adj
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    if totallen == 11:
                        return peak_data_temp.reset_index(drop=True)
                    else:
                        for i in range(11, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp.reset_index(drop=True)

            else:

                if return_whole == False:
                    return None
                else:
                    peak_data_temp = peak_data_temp.drop(index)
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    return peak_data_temp.reset_index(drop=True)

    elif method == "Blockify":

        if type(background) == pd.DataFrame:

            backgroundcounts = len(
                background[
                    (background["Chr"] == chrom1)
                    & (background["Start"] >= start - length)
                    & (background["Start"] <= end)
                ]
            )

            expdatacounts_lam = len(expdata[(expdata["Chr"] == chrom1)])
            backgroundcounts_lam = len(background[(background["Chr"] == chrom1)])
            expinsertion_background = expdatacounts_lam * (
                backgroundcounts / backgroundcounts_lam
            )

            if test_method == "poisson":
                pvalue = _compute_cumulative_poisson(
                    expdatacounts,
                    backgroundcounts,
                    expdatacounts_lam,
                    backgroundcounts_lam,
                    pseudocounts,
                )
            elif test_method == "binomial":
                pvalue = binom_test(
                    int(expdatacounts + pseudocounts),
                    n=expdatacounts_lam,
                    p=((backgroundcounts + pseudocounts) / backgroundcounts_lam),
                    alternative="greater",
                ).pvalue

            pvalue_adj = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue)
            ]

            if (pvalue_cutoffbg == None) or (pvalue <= float(pvalue_cutoffbg or 0)):

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom1,
                                start,
                                end,
                                expdatacounts,
                                backgroundcounts,
                                expinsertion_background,
                                pvalue,
                                pvalue_adj,
                            ]
                        ],
                        columns=index_list,
                    )
                else:
                    peak_data_temp.iloc[index, 2] = end
                    peak_data_temp.iloc[index, 3] = expdatacounts
                    peak_data_temp.iloc[index, 4] = backgroundcounts
                    peak_data_temp.iloc[index, 5] = expinsertion_background
                    peak_data_temp.iloc[index, 6] = pvalue
                    peak_data_temp.iloc[index, 7] = pvalue_adj
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    if totallen == 8:
                        return peak_data_temp.reset_index(drop=True)
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp.reset_index(drop=True)

            else:

                if return_whole == False:
                    return None
                else:
                    peak_data_temp = peak_data_temp.drop(index)
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    return peak_data_temp.reset_index(drop=True)

        else:

            TTAAcounts = len(
                TTAA_data[
                    (TTAA_data[0] == chrom1)
                    & (TTAA_data[2] >= start - length)
                    & (TTAA_data[2] <= end)
                ]
            )

            expdatacounts_lam = len(expdata[(expdata["Chr"] == chrom1)])
            TTAAcounts_lam = len(TTAA_data[(TTAA_data[0] == chrom1)])
            expinsertion_TTAA = expdatacounts_lam * (TTAAcounts / TTAAcounts_lam)

            if test_method == "poisson":
                pvalue = _compute_cumulative_poisson(
                    expdatacounts,
                    TTAAcounts,
                    expdatacounts_lam,
                    TTAAcounts_lam,
                    pseudocounts,
                )
            elif test_method == "binomial":
                pvalue = binom_test(
                    int(expdatacounts + pseudocounts),
                    n=expdatacounts_lam,
                    p=((TTAAcounts + pseudocounts) / TTAAcounts_lam),
                    alternative="greater",
                ).pvalue

            pvalue_adj = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue)
            ]

            if (pvalue_cutoff == None) or (pvalue <= float(pvalue_cutoff or 0)):

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom1,
                                start,
                                end,
                                expdatacounts,
                                TTAAcounts,
                                expinsertion_TTAA,
                                pvalue,
                                pvalue_adj,
                            ]
                        ],
                        columns=index_list,
                    )
                else:
                    peak_data_temp.iloc[index, 2] = end
                    peak_data_temp.iloc[index, 3] = expdatacounts
                    peak_data_temp.iloc[index, 4] = TTAAcounts
                    peak_data_temp.iloc[index, 5] = expinsertion_TTAA
                    peak_data_temp.iloc[index, 6] = pvalue
                    peak_data_temp.iloc[index, 7] = pvalue_adj
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    if totallen == 8:
                        return peak_data_temp.reset_index(drop=True)
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp.reset_index(drop=True)

            else:

                if return_whole == False:
                    return None
                else:
                    peak_data_temp = peak_data_temp.drop(index)
                    peak_data_temp = peak_data_temp.drop(index + 1)

                    return peak_data_temp.reset_index(drop=True)

    else:

        raise ValueError("Not valid Method.")


def separate_peaks(
    peak_data: pd.DataFrame,
    index: int,
    middle_start: int,
    middle_end: int,
    expdata: Optional[pd.DataFrame] = None,
    background: Optional[pd.DataFrame] = None,
    method: _Peakcalling_Method = "CCcaller",
    reference: _reference = "hg38",
    test_method: _Peakcalling_Method = "poisson",
    lam_win_size: Optional[int] = 100000,
    pvalue_cutoff: Optional[float] = None,
    pvalue_cutoffbg: Optional[float] = None,
    pvalue_cutoffTTAA: Optional[float] = None,
    pseudocounts: float = 0.2,
    return_whole: bool = False,
):

    """\
    Separate two peaks.

    This function separate one peak into two.

    :param peak_data:
        pd.DataFrame for peak data. Please input the original data from call_peaks function.
    :param index
        The index for the peak to separate.
    :param middle_start
        The start point of the cutoff which is the end point of the first peak after separation.
    :param middle_end
        TThe end point of the cutoff which is the start point of the second peak after separation.
    :param expdata:
        pd.DataFrame with the first three columns as chromosome, start and end.
    :param background:
        pd.DataFrame with the first three columns as chromosome, start and end.
    :param method:
        `'CCcaller'` is a method considering the maxdistance between insertions in the data,
        `'cc_tools'` uses the idea adapted from :cite:`zhang2008model` and
        `here <https://hbctraining.github.io/Intro-to-ChIPseq/lessons/05_peak_calling_macs.html>`__.
        `'Blockify'` uses the method from :cite:`moudgil2020self` and `here <https://blockify.readthedocs.io/en/latest/>`__.
    :param reference:
        We currently have `'hg38'` for human data, `'mm10'` for mouse data and `'sacCer3'` for yeast data.
    :param pvalue_cutoff:
        The P-value cutoff for a backgound free situation. If None, no filteration.
    :param pvalue_cutoffbg:
        The P-value cutoff for backgound data when backgound exists. If None, no filteration.
    :param pvalue_cutoffTTAA:
        The P-value cutoff for reference data when backgound exists.
        Note that pvalue_cutoffTTAA is recommended to be lower than pvalue_cutoffbg. If None, no filteration.
    :param pseudocounts:
        Number for pseudocounts added for the pyhothesis.
    :param return_whole:
        If False, return only the combined peak.
        If True, return the whole peak dataframe.

    :Examples:
    >>> import pycallingcards as cc
    >>> qbed_data = cc.datasets.mousecortex_data(data="qbed")
    >>> peak_data = cc.pp.call_peaks(qbed_data, method = "CCcaller", reference = "mm10",  maxbetween = 2000,pvalue_cutoff = 0.01, pseudocounts = 1, record = True)
    >>> cc.pp.separate_peaks(peak_data,1,4807673,4808049,expdata=qbed_data,reference='mm10',method = "CCcaller",test_method='poisson',pvalue_cutoff=0.01,pseudocounts=0.1,return_whole=False)


    """

    peak_data_temp = peak_data.copy()

    if test_method == "poisson":
        from scipy.stats import poisson
    elif test_method == "binomial":
        from scipy.stats import binom_test

    chrom = peak_data_temp.iloc[index, 0]
    start = peak_data_temp.iloc[index, 1]
    end = peak_data_temp.iloc[index, 2]

    index_list = list(peak_data_temp.columns)
    totallen = len(index_list)

    if totallen == 3:

        if return_whole == False:
            return pd.DataFrame(
                [[chrom, start, middle_start], [chrom, middle_end, end]],
                columns=index_list,
            )
        else:
            peak_1 = peak_data_temp.iloc[
                : index + 1,
            ].copy()
            peak_2 = peak_data_temp.iloc[
                index:,
            ].copy()
            peak_1.iloc[index, 2] = middle_start
            peak_2.iloc[0, 1] = middle_end
            peak_data_temp = pd.concat([peak_1, peak_2], ignore_index=True)
            return peak_data_temp.reset_index(drop=True)

    if reference == "hg38":
        TTAA_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_hg38_ccf.bed",
            sep="\t",
            header=None,
        )
        length = 4

    elif reference == "mm10":
        TTAA_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/TTAA_mm10_ccf.bed",
            sep="\t",
            header=None,
        )
        length = 4

    elif reference == "sacCer3":
        TTAA_data = pd.read_csv(
            "https://github.com/The-Mitra-Lab/pycallingcards_data/releases/download/data/yeast_Background.ccf",
            sep="\t",
            header=None,
        )
        length = 1

    if type(expdata) == pd.DataFrame:
        expdatacounts1 = len(
            expdata[
                (expdata["Chr"] == chrom)
                & (expdata["Start"] >= start - length)
                & (expdata["Start"] <= middle_start)
            ]
        )
        expdatacounts2 = len(
            expdata[
                (expdata["Chr"] == chrom)
                & (expdata["Start"] >= middle_end - length)
                & (expdata["Start"] <= end)
            ]
        )
    else:
        raise Exception("Please input expdata.")

    print(
        "The new pvalue_adj is not the true one. It searches for the cloest one among the data."
    )

    if method == "CCcaller":

        TTAAcounts1 = len(
            TTAA_data[
                (TTAA_data[0] == chrom)
                & (TTAA_data[2] >= start - length)
                & (TTAA_data[2] <= middle_start)
            ]
        )
        TTAAcounts2 = len(
            TTAA_data[
                (TTAA_data[0] == chrom)
                & (TTAA_data[2] >= middle_end - length)
                & (TTAA_data[2] <= end)
            ]
        )

        if lam_win_size == None:
            expdatacounts_lam1 = len(expdata[(expdata["Chr"] == chrom)])
            TTAAcounts_lam1 = len(TTAA_data[(TTAA_data[0] == chrom)])
            expdatacounts_lam2 = expdatacounts_lam1
            TTAAcounts_lam2 = TTAAcounts_lam1
        else:

            expdatacounts_lam1 = len(
                expdata[
                    (expdata["Chr"] == chrom)
                    & (expdata["Start"] >= start - length - lam_win_size / 2 + 1)
                    & (expdata["Start"] <= middle_start + lam_win_size / 2)
                ]
            )
            TTAAcounts_lam1 = len(
                TTAA_data[
                    (TTAA_data[0] == chrom)
                    & (TTAA_data[2] >= start - length - lam_win_size / 2 + 1)
                    & (TTAA_data[2] <= middle_start + lam_win_size / 2)
                ]
            )

            expdatacounts_lam2 = len(
                expdata[
                    (expdata["Chr"] == chrom)
                    & (expdata["Start"] >= middle_end - length - lam_win_size / 2 + 1)
                    & (expdata["Start"] <= end + lam_win_size / 2)
                ]
            )
            TTAAcounts_lam2 = len(
                TTAA_data[
                    (TTAA_data[0] == chrom)
                    & (TTAA_data[2] >= middle_end - length - lam_win_size / 2 + 1)
                    & (TTAA_data[2] <= end + lam_win_size / 2)
                ]
            )

        expinsertion_TTAA1 = expdatacounts_lam1 * (TTAAcounts1 / TTAAcounts_lam1)
        expinsertion_TTAA2 = expdatacounts_lam2 * (TTAAcounts2 / TTAAcounts_lam2)

        if test_method == "poisson":

            pvalue1 = _compute_cumulative_poisson(
                expdatacounts1,
                TTAAcounts1,
                expdatacounts_lam1,
                TTAAcounts_lam1,
                pseudocounts,
            )
            pvalue2 = _compute_cumulative_poisson(
                expdatacounts2,
                TTAAcounts2,
                expdatacounts_lam2,
                TTAAcounts_lam2,
                pseudocounts,
            )

        elif test_method == "binomial":

            pvalue1 = binom_test(
                int(expdatacounts1 + pseudocounts),
                n=expdatacounts_lam1,
                p=((TTAAcounts1 + pseudocounts) / TTAAcounts_lam1),
                alternative="greater",
            ).pvalue
            pvalue1 = binom_test(
                int(expdatacounts2 + pseudocounts),
                n=expdatacounts_lam2,
                p=((TTAAcounts2 + pseudocounts) / TTAAcounts_lam2),
                alternative="greater",
            ).pvalue

        if type(background) == pd.DataFrame:

            backgroundcounts1 = len(
                background[
                    (background["Chr"] == chrom)
                    & (background["Start"] >= start - length)
                    & (background["Start"] <= middle_start)
                ]
            )
            backgroundcounts2 = len(
                background[
                    (background["Chr"] == chrom)
                    & (background["Start"] >= middle_end - length)
                    & (background["Start"] <= end)
                ]
            )

            if lam_win_size == None:
                backgroundcounts_lam1 = len(background[(background["Chr"] == chrom)])
                backgroundcounts_lam2 = backgroundcounts_lam1
            else:
                backgroundcounts_lam1 = len(
                    background[
                        (background["Chr"] == chrom)
                        & (background["Start"] >= start - length - lam_win_size / 2 + 1)
                        & (background["Start"] <= middle_start + lam_win_size / 2)
                    ]
                )
                backgroundcounts_lam2 = len(
                    background[
                        (background["Chr"] == chrom)
                        & (
                            background["Start"]
                            >= middle_end - length - lam_win_size / 2 + 1
                        )
                        & (background["Start"] <= end + lam_win_size / 2)
                    ]
                )

            expinsertion_bg1 = expdatacounts_lam1 * (
                backgroundcounts1 / backgroundcounts_lam1
            )
            expinsertion_bg2 = expdatacounts_lam2 * (
                backgroundcounts2 / backgroundcounts_lam2
            )

            if test_method == "poisson":

                pvalue_bg1 = _compute_cumulative_poisson(
                    expdatacounts1,
                    backgroundcounts1,
                    expdatacounts_lam1,
                    backgroundcounts_lam1,
                    pseudocounts,
                )
                pvalue_bg2 = _compute_cumulative_poisson(
                    expdatacounts2,
                    backgroundcounts2,
                    expdatacounts_lam2,
                    backgroundcounts_lam2,
                    pseudocounts,
                )

            elif test_method == "binomial":

                pvalue_bg1 = binom_test(
                    int(expdatacounts1 + pseudocounts),
                    n=expdatacounts_lam1,
                    p=((backgroundcounts1 + pseudocounts) / backgroundcounts_lam1),
                    alternative="greater",
                ).pvalue
                pvalue_bg2 = binom_test(
                    int(expdatacounts2 + pseudocounts),
                    n=expdatacounts_lam2,
                    p=((backgroundcounts2 + pseudocounts) / backgroundcounts_lam2),
                    alternative="greater",
                ).pvalue

            pvalue_adj1 = peak_data["pvalue_adj Reference"][
                _closest(list(peak_data["pvalue Reference"]), pvalue1)
            ]
            # pvalue_adj_bg1 = peak_data["pvalue_adj Background"][_closest(list(peak_data["pvalue Background"]), pvalue_bg1)]
            pvalue_adj2 = peak_data["pvalue_adj Reference"][
                _closest(list(peak_data["pvalue Reference"]), pvalue2)
            ]
            # pvalue_adj_bg2 = peak_data["pvalue_adj Background"][_closest(list(peak_data["pvalue Background"]), pvalue_bg2)]

            condition1 = (
                (pvalue_cutoffTTAA == None)
                or (pvalue1 <= float(pvalue_cutoffTTAA or 0))
            ) and (
                (pvalue_cutoffbg == None) or (pvalue_bg1 <= float(pvalue_cutoffbg or 0))
            )
            condition2 = (
                (pvalue_cutoffTTAA == None)
                or (pvalue2 <= float(pvalue_cutoffTTAA or 0))
            ) and (
                (pvalue_cutoffbg == None) or (pvalue_bg2 <= float(pvalue_cutoffbg or 0))
            )

            if condition1 and condition2:

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                expdatacounts1,
                                backgroundcounts1,
                                TTAAcounts1,
                                expinsertion_bg1,
                                expinsertion_TTAA1,
                                pvalue_bg1,
                                pvalue1,
                                pvalue_adj1,
                            ],
                            [
                                chrom,
                                middle_end,
                                end,
                                expdatacounts2,
                                backgroundcounts2,
                                TTAAcounts2,
                                expinsertion_bg2,
                                expinsertion_TTAA2,
                                pvalue_bg2,
                                pvalue2,
                                pvalue_adj2,
                            ],
                        ],
                        columns=index_list,
                    )
                else:

                    peak_1 = peak_data_temp.iloc[
                        : index + 1,
                    ].copy()
                    peak_2 = peak_data_temp.iloc[
                        index:,
                    ].copy()
                    peak_1.iloc[index, 2] = middle_start
                    peak_1.iloc[index, 3] = expdatacounts1
                    peak_1.iloc[index, 4] = backgroundcounts1
                    peak_1.iloc[index, 5] = TTAAcounts1
                    peak_1.iloc[index, 6] = expinsertion_bg1
                    peak_1.iloc[index, 7] = expinsertion_TTAA1
                    peak_1.iloc[index, 8] = pvalue_bg1
                    peak_1.iloc[index, 9] = pvalue1
                    # peak_1.iloc[index,10] = pvalue_adj_bg1
                    peak_1.iloc[index, 10] = pvalue_adj1
                    peak_2.iloc[0, 1] = middle_end
                    peak_2.iloc[0, 3] = expdatacounts2
                    peak_2.iloc[0, 4] = backgroundcounts2
                    peak_2.iloc[0, 5] = TTAAcounts2
                    peak_2.iloc[0, 6] = expinsertion_bg2
                    peak_2.iloc[0, 7] = expinsertion_TTAA2
                    # peak_2.iloc[0,10] = pvalue_adj_bg2
                    peak_2.iloc[0, 10] = pvalue_adj2
                    peak_data_temp = pd.concat([peak_1, peak_2], ignore_index=True)
                    peak_data_temp = peak_data_temp.reset_index(drop=True)

                    if totallen == 11:
                        return peak_data_temp
                    else:
                        for i in range(11, totallen):
                            peak_data_temp.iloc[index, i] = None
                            peak_data_temp.iloc[index + 1, i] = None
                        return peak_data_temp

            elif condition1 and not condition2:

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                expdatacounts1,
                                backgroundcounts1,
                                TTAAcounts1,
                                expinsertion_bg1,
                                expinsertion_TTAA1,
                                pvalue_bg1,
                                pvalue1,
                                pvalue_adj1,
                            ]
                        ],
                        columns=index_list,
                    )
                else:

                    peak_data_temp.iloc[index, 2] = middle_start
                    peak_data_temp.iloc[index, 3] = expdatacounts1
                    peak_data_temp.iloc[index, 4] = backgroundcounts1
                    peak_data_temp.iloc[index, 5] = TTAAcounts1
                    peak_data_temp.iloc[index, 6] = expinsertion_bg1
                    peak_data_temp.iloc[index, 7] = expinsertion_TTAA1
                    peak_data_temp.iloc[index, 8] = pvalue_bg1
                    peak_data_temp.iloc[index, 9] = pvalue1
                    #  peak_data_temp.iloc[index,10] = pvalue_adj_bg1
                    peak_data_temp.iloc[index, 10] = pvalue_adj1

                    if totallen == 11:
                        return peak_data_temp
                    else:
                        for i in range(11, totallen):
                            peak_data_temp.iloc[index, :][i] = None
                        return peak_data_temp

            elif not condition1 and condition2:

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                middle_end,
                                end,
                                expdatacounts2,
                                backgroundcounts2,
                                TTAAcounts2,
                                expinsertion_bg2,
                                expinsertion_TTAA2,
                                pvalue_bg2,
                                pvalue2,
                                pvalue_adj2,
                            ]
                        ],
                        columns=index_list,
                    )
                else:

                    peak_data_temp.iloc[index, 1] = middle_end
                    peak_data_temp.iloc[index, 3] = expdatacounts2
                    peak_data_temp.iloc[index, 4] = backgroundcounts2
                    peak_data_temp.iloc[index, 5] = TTAAcounts2
                    peak_data_temp.iloc[index, 6] = expinsertion_bg2
                    peak_data_temp.iloc[index, 7] = expinsertion_TTAA2
                    peak_data_temp.iloc[index, 8] = pvalue_bg2
                    peak_data_temp.iloc[index, 9] = pvalue2
                    # peak_data_temp.iloc[index,10] = pvalue_adj_bg2
                    peak_data_temp.iloc[index, 10] = pvalue_adj2

                    if totallen == 11:
                        return peak_data_temp
                    else:
                        for i in range(11, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            else:

                if return_whole == False:
                    return None
                else:
                    peak_data_temp = peak_data_temp.drop(index)
                    return peak_data_temp.reset_index(drop=True)

        else:

            pvalue_adj1 = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue1)
            ]
            pvalue_adj2 = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue2)
            ]

            condition1 = (pvalue_cutoff == None) or (
                pvalue1 <= float(pvalue_cutoff or 0)
            )
            condition2 = (pvalue_cutoff == None) or (
                pvalue2 <= float(pvalue_cutoff or 0)
            )

            if condition1 and condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                expdatacounts1,
                                TTAAcounts1,
                                expinsertion_TTAA1,
                                pvalue1,
                                pvalue_adj1,
                            ],
                            [
                                chrom,
                                middle_end,
                                end,
                                expdatacounts2,
                                TTAAcounts2,
                                expinsertion_TTAA2,
                                pvalue2,
                                pvalue_adj2,
                            ],
                        ],
                        columns=index_list,
                    )

                else:

                    peak_1 = peak_data_temp.iloc[
                        : index + 1,
                    ].copy()
                    peak_2 = peak_data_temp.iloc[
                        index:,
                    ].copy()
                    peak_1.iloc[index, 2] = middle_start
                    peak_1.iloc[index, 3] = expdatacounts1
                    peak_1.iloc[index, 4] = TTAAcounts1
                    peak_1.iloc[index, 5] = expinsertion_TTAA1
                    peak_1.iloc[index, 6] = pvalue1
                    peak_1.iloc[index, 7] = pvalue_adj1

                    peak_2.iloc[0, 1] = middle_end
                    peak_2.iloc[0, 3] = expdatacounts2
                    peak_2.iloc[0, 4] = TTAAcounts2
                    peak_2.iloc[0, 5] = expinsertion_TTAA2
                    peak_2.iloc[0, 6] = pvalue2
                    peak_2.iloc[0, 7] = pvalue_adj2

                    peak_data_temp = pd.concat([peak_1, peak_2], ignore_index=True)
                    peak_data_temp = peak_data_temp.reset_index(drop=True)

                    if totallen == 8:
                        return peak_data_temp
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                            peak_data_temp.iloc[index + 1, i] = None
                        return peak_data_temp

            elif condition1 and not condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                expdatacounts1,
                                TTAAcounts1,
                                expinsertion_TTAA1,
                                pvalue1,
                                pvalue_adj1,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 2] = middle_start
                    peak_data_temp.iloc[index, 3] = expdatacounts1
                    peak_data_temp.iloc[index, 4] = TTAAcounts1
                    peak_data_temp.iloc[index, 5] = expinsertion_TTAA1
                    peak_data_temp.iloc[index, 6] = pvalue1
                    peak_data_temp.iloc[index, 7] = pvalue_adj1

                    if totallen == 8:
                        return peak_data_temp
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            elif not condition1 and condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                middle_end,
                                end,
                                expdatacounts2,
                                TTAAcounts2,
                                expinsertion_TTAA2,
                                pvalue2,
                                pvalue_adj2,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 2] = middle_start
                    peak_data_temp.iloc[index, 3] = expdatacounts2
                    peak_data_temp.iloc[index, 4] = TTAAcounts2
                    peak_data_temp.iloc[index, 5] = expinsertion_TTAA2
                    peak_data_temp.iloc[index, 6] = pvalue2
                    peak_data_temp.iloc[index, 7] = pvalue_adj2

                    if totallen == 8:
                        return peak_data_temp
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            else:

                if return_whole == False:
                    return None

                else:
                    peak_data_temp = peak_data_temp.drop(index)
                    return peak_data_temp.reset_index(drop=True)

    elif method == "cc_tools":

        multinumber = 100000000
        sumcount_expdata = len(expdata)
        TTAAcounts1 = len(
            TTAA_data[
                (TTAA_data[0] == chrom)
                & (TTAA_data[2] >= start - length)
                & (TTAA_data[2] <= middle_start)
            ]
        )
        TTAAcounts2 = len(
            TTAA_data[
                (TTAA_data[0] == chrom)
                & (TTAA_data[2] >= middle_end - length)
                & (TTAA_data[2] <= end)
            ]
        )

        if lam_win_size == None:
            expdatacounts_lam1 = len(expdata[(expdata["Chr"] == chrom)])
            TTAAcounts_lam1 = len(TTAA_data[(TTAA_data[0] == chrom)])
            expdatacounts_lam2 = expdatacounts_lam1
            TTAAcounts_lam2 = TTAAcounts_lam1
        else:
            expdatacounts_lam1 = len(
                expdata[
                    (expdata["Chr"] == chrom)
                    & (expdata["Start"] >= start - length - lam_win_size / 2 + 1)
                    & (expdata["Start"] <= middle_start + lam_win_size / 2)
                ]
            )
            TTAAcounts_lam1 = len(
                TTAA_data[
                    (TTAA_data[0] == chrom)
                    & (TTAA_data[2] >= start - length - lam_win_size / 2 + 1)
                    & (TTAA_data[2] <= middle_start + lam_win_size / 2)
                ]
            )
            expdatacounts_lam2 = len(
                expdata[
                    (expdata["Chr"] == chrom)
                    & (expdata["Start"] >= middle_end - length - lam_win_size / 2 + 1)
                    & (expdata["Start"] <= end + lam_win_size / 2)
                ]
            )
            TTAAcounts_lam2 = len(
                TTAA_data[
                    (TTAA_data[0] == chrom)
                    & (TTAA_data[2] >= middle_end - length - lam_win_size / 2 + 1)
                    & (TTAA_data[2] <= end + lam_win_size / 2)
                ]
            )

        expinsertion_TTAA1 = expdatacounts_lam1 * (TTAAcounts1 / TTAAcounts_lam1)
        counts_median1 = np.median(
            list(
                expdata[
                    (expdata["Chr"] == chrom)
                    & (expdata["Start"] >= start - length)
                    & (expdata["Start"] <= middle_start)
                ].iloc[:, 1]
            )
        )
        expinsertion_TTAA2 = expdatacounts_lam2 * (TTAAcounts2 / TTAAcounts_lam2)
        counts_median2 = np.median(
            list(
                expdata[
                    (expdata["Chr"] == chrom)
                    & (expdata["Start"] >= middle_end - length)
                    & (expdata["Start"] <= end)
                ].iloc[:, 1]
            )
        )

        TPH1 = float(expdatacounts1) * multinumber / sumcount_expdata
        frac_exp1 = float(expdatacounts1) / sumcount_expdata
        TPH2 = float(expdatacounts2) * multinumber / sumcount_expdata
        frac_exp2 = float(expdatacounts2) / sumcount_expdata

        if test_method == "poisson":

            pvalue1 = _compute_cumulative_poisson(
                expdatacounts1,
                TTAAcounts1,
                expdatacounts_lam1,
                TTAAcounts_lam1,
                pseudocounts,
            )
            pvalue2 = _compute_cumulative_poisson(
                expdatacounts2,
                TTAAcounts2,
                expdatacounts_lam2,
                TTAAcounts_lam2,
                pseudocounts,
            )

        elif test_method == "binomial":

            pvalue1 = binom_test(
                int(expdatacounts1 + pseudocounts),
                n=expdatacounts_lam1,
                p=((TTAAcounts1 + pseudocounts) / TTAAcounts_lam1),
                alternative="greater",
            ).pvalue
            pvalue2 = binom_test(
                int(expdatacounts2 + pseudocounts),
                n=expdatacounts_lam2,
                p=((TTAAcounts2 + pseudocounts) / TTAAcounts_lam2),
                alternative="greater",
            ).pvalue

        if type(background) == pd.DataFrame:

            backgroundcounts1 = len(
                background[
                    (background["Chr"] == chrom)
                    & (background["Start"] >= start - length)
                    & (background["Start"] <= middle_start)
                ]
            )
            backgroundcounts2 = len(
                background[
                    (background["Chr"] == chrom)
                    & (background["Start"] >= middle_end - length)
                    & (background["Start"] <= end)
                ]
            )

            sumcount_background = len(background)

            if lam_win_size == None:
                backgroundcounts_lam1 = len(background[(background["Chr"] == chrom)])
                backgroundcounts_lam2 = backgroundcounts_lam1
            else:
                backgroundcounts_lam1 = len(
                    background[
                        (background["Chr"] == chrom)
                        & (background["Start"] >= start - length - lam_win_size / 2 + 1)
                        & (background["Start"] <= middle_start + lam_win_size / 2)
                    ]
                )
                backgroundcounts_lam2 = len(
                    background[
                        (background["Chr"] == chrom)
                        & (
                            background["Start"]
                            >= middle_end - length - lam_win_size / 2 + 1
                        )
                        & (background["Start"] <= end + lam_win_size / 2)
                    ]
                )

            expinsertion_bg1 = expdatacounts_lam1 * (
                backgroundcounts1 / backgroundcounts_lam1
            )
            TPH_bg1 = float(backgroundcounts1) * multinumber / sumcount_background
            frac_exp_bg1 = float(backgroundcounts1) / sumcount_background
            expinsertion_bg2 = expdatacounts_lam2 * (
                backgroundcounts2 / backgroundcounts_lam2
            )
            TPH_bg2 = float(backgroundcounts2) * multinumber / sumcount_background
            frac_exp_bg2 = float(backgroundcounts2) / sumcount_background

            if test_method == "poisson":

                pvalue_bg1 = _compute_cumulative_poisson(
                    expdatacounts1,
                    backgroundcounts1,
                    expdatacounts_lam1,
                    backgroundcounts_lam1,
                    pseudocounts,
                )
                pvalue_bg2 = _compute_cumulative_poisson(
                    expdatacounts2,
                    backgroundcounts2,
                    expdatacounts_lam2,
                    backgroundcounts_lam2,
                    pseudocounts,
                )

            elif test_method == "binomial":

                pvalue_bg1 = binom_test(
                    int(expdatacounts1 + pseudocounts),
                    n=expdatacounts_lam1,
                    p=((backgroundcounts1 + pseudocounts) / backgroundcounts_lam1),
                    alternative="greater",
                ).pvalue
                pvalue_bg2 = binom_test(
                    int(expdatacounts2 + pseudocounts),
                    n=expdatacounts_lam2,
                    p=((backgroundcounts2 + pseudocounts) / backgroundcounts_lam2),
                    alternative="greater",
                ).pvalue

            pvalue_adj1 = peak_data["pvalue_adj Reference"][
                _closest(list(peak_data["pvalue Reference"]), pvalue1)
            ]
            #  pvalue_adj_bg1 = peak_data["pvalue_adj Background"][_closest(list(peak_data["pvalue Background"]), pvalue_bg1)]
            pvalue_adj2 = peak_data["pvalue_adj Reference"][
                _closest(list(peak_data["pvalue Reference"]), pvalue2)
            ]
            #  pvalue_adj_bg2 = peak_data["pvalue_adj Background"][_closest(list(peak_data["pvalue Background"]), pvalue_bg2)]

            condition1 = (
                (pvalue_cutoffTTAA == None)
                or (pvalue1 <= float(pvalue_cutoffTTAA or 0))
            ) and (
                (pvalue_cutoffbg == None) or (pvalue_bg1 <= float(pvalue_cutoffbg or 0))
            )
            condition2 = (
                (pvalue_cutoffTTAA == None)
                or (pvalue2 <= float(pvalue_cutoffTTAA or 0))
            ) and (
                (pvalue_cutoffbg == None) or (pvalue_bg2 <= float(pvalue_cutoffbg or 0))
            )

            if condition1 and condition2:

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                counts_median1,
                                expdatacounts1,
                                backgroundcounts1,
                                TTAAcounts1,
                                pvalue1,
                                pvalue_bg1,
                                frac_exp1,
                                TPH1,
                                frac_exp_bg1,
                                TPH_bg1,
                                frac_exp1 - frac_exp_bg1,
                                pvalue_adj1,
                            ],
                            [
                                chrom,
                                middle_end,
                                end,
                                counts_median2,
                                expdatacounts2,
                                backgroundcounts2,
                                TTAAcounts2,
                                pvalue2,
                                pvalue_bg2,
                                frac_exp2,
                                TPH2,
                                frac_exp_bg2,
                                TPH_bg2,
                                frac_exp2 - frac_exp_bg2,
                                pvalue_adj2,
                            ],
                        ],
                        columns=index_list,
                    )
                else:

                    peak_1 = peak_data_temp.iloc[
                        : index + 1,
                    ].copy()
                    peak_2 = peak_data_temp.iloc[
                        index:,
                    ].copy()
                    peak_1.iloc[index, 2] = middle_start
                    peak_1.iloc[index, 3] = counts_median1
                    peak_1.iloc[index, 4] = expdatacounts1
                    peak_1.iloc[index, 5] = backgroundcounts1
                    peak_1.iloc[index, 6] = TTAAcounts1
                    peak_1.iloc[index, 7] = pvalue1
                    peak_1.iloc[index, 8] = pvalue_bg1
                    peak_1.iloc[index, 9] = frac_exp1
                    peak_1.iloc[index, 10] = TPH1
                    peak_1.iloc[index, 11] = frac_exp_bg1
                    peak_1.iloc[index, 12] = TPH_bg1
                    peak_1.iloc[index, 13] = frac_exp1 - frac_exp_bg1
                    #  peak_1.iloc[index,14] = pvalue_adj_bg1
                    peak_1.iloc[index, 14] = pvalue_adj1

                    peak_2.iloc[index, 1] = middle_end
                    peak_2.iloc[index, 3] = counts_median2
                    peak_2.iloc[index, 4] = expdatacounts2
                    peak_2.iloc[index, 5] = backgroundcounts2
                    peak_2.iloc[index, 6] = TTAAcounts2
                    peak_2.iloc[index, 7] = pvalue2
                    peak_2.iloc[index, 8] = pvalue_bg2
                    peak_2.iloc[index, 9] = frac_exp2
                    peak_2.iloc[index, 10] = TPH2
                    peak_2.iloc[index, 11] = frac_exp_bg2
                    peak_2.iloc[index, 12] = TPH_bg2
                    peak_2.iloc[index, 13] = frac_exp2 - frac_exp_bg2
                    #   peak_2.iloc[index,14] = pvalue_adj_bg2
                    peak_2.iloc[index, 14] = pvalue_adj2

                    peak_data_temp = pd.concat([peak_1, peak_2], ignore_index=True)
                    peak_data_temp = peak_data_temp.reset_index(drop=True)

                    if totallen == 15:
                        return peak_data_temp
                    else:
                        for i in range(15, totallen):
                            peak_data_temp.iloc[index, i] = None
                            peak_data_temp.iloc[index + 1, i] = None
                        return peak_data_temp

            elif condition1 and not condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                counts_median1,
                                expdatacounts1,
                                backgroundcounts1,
                                TTAAcounts1,
                                pvalue1,
                                pvalue_bg1,
                                frac_exp1,
                                TPH1,
                                frac_exp_bg1,
                                TPH_bg1,
                                frac_exp1 - frac_exp_bg1,
                                pvalue_adj1,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 2] = middle_start
                    peak_data_temp.iloc[index, 3] = counts_median1
                    peak_data_temp.iloc[index, 4] = expdatacounts1
                    peak_data_temp.iloc[index, 5] = backgroundcounts1
                    peak_data_temp.iloc[index, 6] = TTAAcounts1
                    peak_data_temp.iloc[index, 7] = pvalue1
                    peak_data_temp.iloc[index, 8] = pvalue_bg1
                    peak_data_temp.iloc[index, 9] = frac_exp1
                    peak_data_temp.iloc[index, 10] = TTAAcounts1
                    peak_data_temp.iloc[index, 11] = frac_exp_bg1
                    peak_data_temp.iloc[index, 12] = TPH_bg1
                    peak_data_temp.iloc[index, 13] = frac_exp1 - frac_exp_bg1
                    #  peak_data_temp.iloc[index,14] = pvalue_adj_bg1
                    peak_data_temp.iloc[index, 14] = pvalue_adj1

                    if totallen == 15:
                        return peak_data_temp
                    else:
                        for i in range(15, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            elif not condition1 and condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                middle_end,
                                end,
                                counts_median2,
                                expdatacounts2,
                                backgroundcounts2,
                                TTAAcounts2,
                                pvalue2,
                                pvalue_bg2,
                                frac_exp2,
                                TPH2,
                                frac_exp_bg2,
                                TPH_bg2,
                                frac_exp2 - frac_exp_bg2,
                                pvalue_adj2,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 1] = middle_end
                    peak_data_temp.iloc[index, 3] = counts_median2
                    peak_data_temp.iloc[index, 4] = expdatacounts2
                    peak_data_temp.iloc[index, 5] = backgroundcounts2
                    peak_data_temp.iloc[index, 6] = TTAAcounts2
                    peak_data_temp.iloc[index, 7] = pvalue2
                    peak_data_temp.iloc[index, 8] = pvalue_bg2
                    peak_data_temp.iloc[index, 9] = frac_exp2
                    peak_data_temp.iloc[index, 10] = TTAAcounts2
                    peak_data_temp.iloc[index, 11] = frac_exp_bg2
                    peak_data_temp.iloc[index, 12] = TPH_bg2
                    peak_data_temp.iloc[index, 13] = frac_exp2 - frac_exp_bg2
                    #   peak_data_temp.iloc[index,14] = pvalue_adj_bg2
                    peak_data_temp.iloc[index, 14] = pvalue_adj2

                    if totallen == 15:
                        return peak_data_temp
                    else:
                        for i in range(15, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            else:

                if return_whole == False:
                    return None

                else:
                    peak_data_temp = peak_data_temp.drop(index)

                    return peak_data_temp

        else:

            pvalue_adj1 = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue1)
            ]
            pvalue_adj2 = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue2)
            ]

            condition1 = (pvalue_cutoff == None) or (
                pvalue1 <= float(pvalue_cutoff or 0)
            )
            condition2 = (pvalue_cutoff == None) or (
                pvalue2 <= float(pvalue_cutoff or 0)
            )

            if condition1 and condition2:

                if return_whole == False:
                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                counts_median1,
                                pvalue1,
                                expdatacounts1,
                                TTAAcounts1,
                                frac_exp1,
                                TPH1,
                                expinsertion_TTAA1,
                                pvalue_adj1,
                            ],
                            [
                                chrom,
                                middle_end,
                                end,
                                counts_median2,
                                pvalue2,
                                expdatacounts2,
                                TTAAcounts2,
                                frac_exp2,
                                TPH2,
                                expinsertion_TTAA2,
                                pvalue_adj2,
                            ],
                        ],
                        columns=index_list,
                    )
                else:

                    peak_1 = peak_data_temp.iloc[
                        : index + 1,
                    ].copy()
                    peak_2 = peak_data_temp.iloc[
                        index:,
                    ].copy()
                    peak_1.iloc[index, 2] = middle_start
                    peak_1.iloc[index, 3] = counts_median1
                    peak_1.iloc[index, 4] = pvalue1
                    peak_1.iloc[index, 5] = expdatacounts1
                    peak_1.iloc[index, 6] = TTAAcounts1
                    peak_1.iloc[index, 7] = frac_exp1
                    peak_1.iloc[index, 8] = TPH1
                    peak_1.iloc[index, 9] = expinsertion_TTAA1
                    peak_1.iloc[index, 10] = pvalue_adj1

                    peak_2.iloc[index, 1] = middle_end
                    peak_2.iloc[index, 3] = counts_median2
                    peak_2.iloc[index, 4] = pvalue2
                    peak_2.iloc[index, 5] = expdatacounts2
                    peak_2.iloc[index, 6] = TTAAcounts2
                    peak_2.iloc[index, 7] = frac_exp2
                    peak_2.iloc[index, 8] = TPH2
                    peak_2.iloc[index, 9] = expinsertion_TTAA2
                    peak_2.iloc[index, 10] = pvalue_adj2

                    peak_data_temp = pd.concat([peak_1, peak_2], ignore_index=True)
                    peak_data_temp = peak_data_temp.reset_index(drop=True)

                    if totallen == 11:
                        return peak_data_temp.reset_index(drop=True)
                    else:
                        for i in range(11, totallen):
                            peak_data_temp.iloc[index, i] = None
                            peak_data_temp.iloc[index + 1, i] = None
                        return peak_data_temp.reset_index(drop=True)

            elif condition1 and not condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                counts_median1,
                                pvalue1,
                                expdatacounts1,
                                TTAAcounts1,
                                frac_exp1,
                                TPH1,
                                expinsertion_TTAA1,
                                pvalue_adj1,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 2] = middle_start
                    peak_data_temp.iloc[index, 3] = counts_median1
                    peak_data_temp.iloc[index, 4] = pvalue1
                    peak_data_temp.iloc[index, 5] = expdatacounts1
                    peak_data_temp.iloc[index, 6] = TTAAcounts1
                    peak_data_temp.iloc[index, 7] = frac_exp1
                    peak_data_temp.iloc[index, 8] = TPH1
                    peak_data_temp.iloc[index, 9] = expinsertion_TTAA1
                    peak_data_temp.iloc[index, 10] = pvalue_adj1

                    if totallen == 11:
                        return peak_data_temp
                    else:
                        for i in range(11, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            elif not condition1 and condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                middle_end,
                                end,
                                counts_median2,
                                pvalue2,
                                expdatacounts2,
                                TTAAcounts2,
                                frac_exp2,
                                TPH2,
                                expinsertion_TTAA2,
                                pvalue_adj2,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 1] = middle_end
                    peak_data_temp.iloc[index, 3] = counts_median2
                    peak_data_temp.iloc[index, 4] = pvalue2
                    peak_data_temp.iloc[index, 5] = expdatacounts2
                    peak_data_temp.iloc[index, 6] = TTAAcounts2
                    peak_data_temp.iloc[index, 7] = frac_exp2
                    peak_data_temp.iloc[index, 8] = TPH2
                    peak_data_temp.iloc[index, 9] = expinsertion_TTAA2
                    peak_data_temp.iloc[index, 10] = pvalue_adj2

                    if totallen == 11:
                        return peak_data_temp
                    else:
                        for i in range(11, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            else:

                if return_whole == False:
                    return None
                else:
                    peak_data_temp = peak_data_temp.drop(index)

                    return peak_data_temp.reset_index(drop=True)

    elif method == "Blockify":

        if type(background) == pd.DataFrame:

            backgroundcounts1 = len(
                background[
                    (background["Chr"] == chrom)
                    & (background["Start"] >= start - length)
                    & (background["Start"] <= middle_start)
                ]
            )
            backgroundcounts2 = len(
                background[
                    (background["Chr"] == chrom)
                    & (background["Start"] >= middle_end - length)
                    & (background["Start"] <= end)
                ]
            )

            expdatacounts_lam = len(expdata[(expdata["Chr"] == chrom)])
            backgroundcounts_lam = len(background[(background["Chr"] == chrom)])
            expinsertion_background1 = expdatacounts_lam * (
                backgroundcounts1 / backgroundcounts_lam
            )
            expinsertion_background2 = expdatacounts_lam * (
                backgroundcounts2 / backgroundcounts_lam
            )

            if test_method == "poisson":

                pvalue1 = _compute_cumulative_poisson(
                    expdatacounts1,
                    backgroundcounts1,
                    expdatacounts_lam,
                    backgroundcounts_lam,
                    pseudocounts,
                )
                pvalue2 = _compute_cumulative_poisson(
                    expdatacounts2,
                    backgroundcounts2,
                    expdatacounts_lam,
                    backgroundcounts_lam,
                    pseudocounts,
                )

            elif test_method == "binomial":

                pvalue1 = binom_test(
                    int(expdatacounts1 + pseudocounts),
                    n=expdatacounts_lam,
                    p=((backgroundcounts1 + pseudocounts) / backgroundcounts_lam),
                    alternative="greater",
                ).pvalue
                pvalue2 = binom_test(
                    int(expdatacounts2 + pseudocounts),
                    n=expdatacounts_lam,
                    p=((backgroundcounts2 + pseudocounts) / backgroundcounts_lam),
                    alternative="greater",
                ).pvalue

            pvalue_adj1 = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue1)
            ]
            pvalue_adj2 = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue2)
            ]

            condition1 = (pvalue_cutoff == None) or (
                pvalue1 <= float(pvalue_cutoff or 0)
            )
            condition2 = (pvalue_cutoff == None) or (
                pvalue2 <= float(pvalue_cutoff or 0)
            )

            if condition1 and condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                expdatacounts1,
                                backgroundcounts1,
                                expinsertion_background1,
                                pvalue1,
                                pvalue_adj1,
                            ],
                            [
                                chrom,
                                middle_end,
                                end,
                                expdatacounts2,
                                backgroundcounts2,
                                expinsertion_background2,
                                pvalue2,
                                pvalue_adj2,
                            ],
                        ],
                        columns=index_list,
                    )

                else:

                    peak_1 = peak_data_temp.iloc[
                        : index + 1,
                    ].copy()
                    peak_2 = peak_data_temp.iloc[
                        index:,
                    ].copy()
                    peak_1.iloc[index, 2] = middle_start
                    peak_1.iloc[index, 3] = expdatacounts1
                    peak_1.iloc[index, 4] = backgroundcounts1
                    peak_1.iloc[index, 5] = expinsertion_background1
                    peak_1.iloc[index, 6] = pvalue1
                    peak_1.iloc[index, 7] = pvalue_adj1

                    peak_2.iloc[index, 1] = middle_end
                    peak_2.iloc[index, 3] = expdatacounts2
                    peak_2.iloc[index, 4] = backgroundcounts2
                    peak_2.iloc[index, 5] = expinsertion_background2
                    peak_2.iloc[index, 6] = pvalue2
                    peak_2.iloc[index, 7] = pvalue_adj2

                    peak_data_temp = pd.concat([peak_1, peak_2], ignore_index=True)
                    peak_data_temp = peak_data_temp.reset_index(drop=True)

                    if totallen == 8:
                        return peak_data_temp.reset_index(drop=True)
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp.reset_index(drop=True)

            if condition1 and not condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                expdatacounts1,
                                backgroundcounts1,
                                expinsertion_background1,
                                pvalue1,
                                pvalue_adj1,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 2] = middle_start
                    peak_data_temp.iloc[index, 3] = expdatacounts1
                    peak_data_temp.iloc[index, 4] = backgroundcounts1
                    peak_data_temp.iloc[index, 5] = expinsertion_background1
                    peak_data_temp.iloc[index, 6] = pvalue1
                    peak_data_temp.iloc[index, 7] = pvalue_adj1

                    if totallen == 8:
                        return peak_data_temp
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            if not condition1 and condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                middle_end,
                                end,
                                expdatacounts2,
                                backgroundcounts2,
                                expinsertion_background2,
                                pvalue2,
                                pvalue_adj2,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 1] = middle_end
                    peak_data_temp.iloc[index, 3] = expdatacounts2
                    peak_data_temp.iloc[index, 4] = backgroundcounts2
                    peak_data_temp.iloc[index, 5] = expinsertion_background2
                    peak_data_temp.iloc[index, 6] = pvalue2
                    peak_data_temp.iloc[index, 7] = pvalue_adj2

                    if totallen == 8:
                        return peak_data_temp
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            else:

                if return_whole == False:
                    return None
                else:
                    peak_data_temp = peak_data_temp.drop(index)

                    return peak_data_temp.reset_index(drop=True)

        else:

            backgroundcounts1 = len(
                TTAA_data[
                    (TTAA_data[0] == chrom)
                    & (TTAA_data[1] >= start - length)
                    & (TTAA_data[1] <= middle_start)
                ]
            )
            backgroundcounts2 = len(
                TTAA_data[
                    (TTAA_data[0] == chrom)
                    & (TTAA_data[1] >= middle_end - length)
                    & (TTAA_data[1] <= end)
                ]
            )

            expdatacounts_lam = len(expdata[(expdata["Chr"] == chrom)])
            backgroundcounts_lam = len(TTAA_data[(TTAA_data[0] == chrom)])
            expinsertion_background1 = expdatacounts_lam * (
                backgroundcounts1 / backgroundcounts_lam
            )
            expinsertion_background2 = expdatacounts_lam * (
                backgroundcounts2 / backgroundcounts_lam
            )

            if test_method == "poisson":

                pvalue1 = _compute_cumulative_poisson(
                    expdatacounts1,
                    backgroundcounts1,
                    expdatacounts_lam,
                    backgroundcounts_lam,
                    pseudocounts,
                )
                pvalue2 = _compute_cumulative_poisson(
                    expdatacounts2,
                    backgroundcounts2,
                    expdatacounts_lam,
                    backgroundcounts_lam,
                    pseudocounts,
                )

            elif test_method == "binomial":

                pvalue1 = binom_test(
                    int(expdatacounts1 + pseudocounts),
                    n=expdatacounts_lam,
                    p=((backgroundcounts1 + pseudocounts) / backgroundcounts_lam),
                    alternative="greater",
                ).pvalue
                pvalue2 = binom_test(
                    int(expdatacounts2 + pseudocounts),
                    n=expdatacounts_lam,
                    p=((backgroundcounts2 + pseudocounts) / backgroundcounts_lam),
                    alternative="greater",
                ).pvalue

            pvalue_adj1 = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue1)
            ]
            pvalue_adj2 = peak_data["pvalue_adj"][
                _closest(list(peak_data["pvalue"]), pvalue2)
            ]

            condition1 = (pvalue_cutoff == None) or (
                pvalue1 <= float(pvalue_cutoff or 0)
            )
            condition2 = (pvalue_cutoff == None) or (
                pvalue2 <= float(pvalue_cutoff or 0)
            )

            if condition1 and condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                expdatacounts1,
                                backgroundcounts1,
                                expinsertion_background1,
                                pvalue1,
                                pvalue_adj1,
                            ],
                            [
                                chrom,
                                middle_end,
                                end,
                                expdatacounts2,
                                backgroundcounts2,
                                expinsertion_background2,
                                pvalue2,
                                pvalue_adj2,
                            ],
                        ],
                        columns=index_list,
                    )

                else:

                    peak_1 = peak_data_temp.iloc[
                        : index + 1,
                    ].copy()
                    peak_2 = peak_data_temp.iloc[
                        index:,
                    ].copy()
                    peak_1.iloc[index, 2] = middle_start
                    peak_1.iloc[index, 3] = expdatacounts1
                    peak_1.iloc[index, 4] = backgroundcounts1
                    peak_1.iloc[index, 5] = expinsertion_background1
                    peak_1.iloc[index, 6] = pvalue1
                    peak_1.iloc[index, 7] = pvalue_adj1

                    peak_2.iloc[index, 1] = middle_end
                    peak_2.iloc[index, 3] = expdatacounts2
                    peak_2.iloc[index, 4] = backgroundcounts2
                    peak_2.iloc[index, 5] = expinsertion_background2
                    peak_2.iloc[index, 6] = pvalue2
                    peak_2.iloc[index, 7] = pvalue_adj2

                    peak_data_temp = pd.concat([peak_1, peak_2], ignore_index=True)
                    peak_data_temp = peak_data_temp.reset_index(drop=True)

                    if totallen == 8:
                        return peak_data_temp.reset_index(drop=True)
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp.reset_index(drop=True)

            if condition1 and not condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                start,
                                middle_start,
                                expdatacounts1,
                                backgroundcounts1,
                                expinsertion_background1,
                                pvalue1,
                                pvalue_adj1,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 2] = middle_start
                    peak_data_temp.iloc[index, 3] = expdatacounts1
                    peak_data_temp.iloc[index, 4] = backgroundcounts1
                    peak_data_temp.iloc[index, 5] = expinsertion_background1
                    peak_data_temp.iloc[index, 6] = pvalue1
                    peak_data_temp.iloc[index, 7] = pvalue_adj1

                    if totallen == 8:
                        return peak_data_temp
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            if not condition1 and condition2:

                if return_whole == False:

                    return pd.DataFrame(
                        [
                            [
                                chrom,
                                middle_end,
                                end,
                                expdatacounts2,
                                backgroundcounts2,
                                expinsertion_background2,
                                pvalue2,
                                pvalue_adj2,
                            ]
                        ],
                        columns=index_list,
                    )

                else:

                    peak_data_temp.iloc[index, 1] = middle_end
                    peak_data_temp.iloc[index, 3] = expdatacounts2
                    peak_data_temp.iloc[index, 4] = backgroundcounts2
                    peak_data_temp.iloc[index, 5] = expinsertion_background2
                    peak_data_temp.iloc[index, 6] = pvalue2
                    peak_data_temp.iloc[index, 7] = pvalue_adj2

                    if totallen == 8:
                        return peak_data_temp
                    else:
                        for i in range(8, totallen):
                            peak_data_temp.iloc[index, i] = None
                        return peak_data_temp

            else:

                if return_whole == False:
                    return None
                else:
                    peak_data_temp = peak_data_temp.drop(index)

                    return peak_data_temp.reset_index(drop=True)

    else:

        raise ValueError("Not valid Method.")


def _fdrcorrection(
    peak_data,
    pvalue_adj_cutoff=0.05,
    reference="mm10",
    pvalue_before="pvalue",
    pvalue_after="pvalue_adj",
):

    if len(peak_data) == 0:
        return peak_data

    peak_data_temp = peak_data.copy()
    pvals = np.array(peak_data_temp[pvalue_before])

    if reference == "mm10":
        total_length = 2730871774
    elif reference == "hg38":
        total_length = 3137300923
    elif reference == "sacCer3":
        total_length = 12157105

    pvals_sortind = np.argsort(pvals)
    pvals_sorted = np.take(pvals, pvals_sortind)
    obs = int(
        total_length / np.array(peak_data_temp["End"] - peak_data_temp["Start"]).mean()
    )
    #  obs = len(pvals)
    ecdffactor = (np.arange(1, obs + 1) / float(obs))[: len(pvals)]

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected > 1] = 1

    pvals_corrected_ = np.empty_like(pvals_corrected)
    pvals_corrected_[pvals_sortind] = pvals_corrected
    peak_data_temp[pvalue_after] = pvals_corrected_

    return peak_data_temp[
        peak_data_temp[pvalue_after] <= pvalue_adj_cutoff
    ].reset_index(drop=True)
