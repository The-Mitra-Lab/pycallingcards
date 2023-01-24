import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def footprint(
    peak_data: pd.DataFrame,
    qbed_data: pd.DataFrame,
    fixed_number: float = 50,
    compress_number: float = 0.5,
    delete_unfound: bool = False,
    resultname: list = ["Chr_footprint", "Start_footprint", "End_footprint"],
    return_bed: bool = False,
) -> pd.DataFrame:

    """\
    Find CC footprint in yeast CC data.

    :param peak_data:
        pd.DataFrame of the peak data with the first three columns as chromosome, start and end.
    :param qbed_data:
        pd.DataFrame of the qbed data where peaks were called from.
    :param fixed_number: Default is 50.
        The minimum length of footprint.
    param compress_number: Default is 0.5.
        How many sd should the footprint conpress. The larger compress_number, the shorter the footprint is.
    :param delete_unfound: Default is `False`.
        If `False`, the point that is not a footprint point would keep the original start and end sites.
        If `True`, the point that is not a footprint point would be [None,None,None].
    :param resultname: Default is [ "Chr_footprint","Start_footprint", "End_footprint"].
        The column name of the final results.
    :param return_bed:Default is `False`.
        If `False`, the function would return the footprint result to the end columns of the data.
        If `True`, the function would return only the footprint results and delete all the none columns.

    :Example:
        Please check `tutorial <https://pycallingcards.readthedocs.io/en/latest/tutorials/notebooks/yeast.html>`for details.
    """

    result = []
    for pean_num in range(len(peak_data)):

        chrm = peak_data.iloc[pean_num][0]

        qbed_chr = qbed_data[qbed_data["Chr"] == chrm]
        X = np.array(
            qbed_chr[
                (qbed_chr["Start"] >= peak_data.iloc[pean_num][1])
                & (qbed_chr["End"] <= peak_data.iloc[pean_num][2])
            ]["Start"]
        )

        if len(X) >= 20:

            X = X.reshape((len(X), 1))
            gm = GaussianMixture(n_components=2, random_state=0).fit(X)

            m1 = gm.means_[0][0]
            d1 = np.sqrt(gm.covariances_)[0][0][0]

            m2 = gm.means_[1][0]
            d2 = np.sqrt(gm.covariances_)[1][0][0]

            if m1 <= m2:

                x1 = m1 + d1 * compress_number
                x2 = m2 - d2 * compress_number

                if (x2 - x1) >= fixed_number:
                    result.append([chrm, int(x1), int(x2 + 1)])
                else:
                    if not delete_unfound:
                        result.append(
                            [
                                chrm,
                                peak_data.iloc[pean_num][1],
                                peak_data.iloc[pean_num][2],
                            ]
                        )
                    else:
                        result.append([None, None, None])

            if m1 >= m2:

                x1 = m1 - d1 * compress_number
                x2 = m2 + d2 * compress_number

                if (x1 - x2) >= fixed_number:
                    result.append([chrm, int(x2), int(x1 + 1)])
                else:
                    if not delete_unfound:
                        result.append(
                            [
                                chrm,
                                peak_data.iloc[pean_num][1],
                                peak_data.iloc[pean_num][2],
                            ]
                        )
                    else:
                        result.append([None, None, None])

        else:
            if not delete_unfound:
                result.append(
                    [chrm, peak_data.iloc[pean_num][1], peak_data.iloc[pean_num][2]]
                )
            else:
                result.append([None, None, None])

    result = pd.DataFrame(result).set_index(peak_data.index)
    result.columns = resultname

    if return_bed:
        result = result.dropna()
        result.iloc[:, 1:3] = result.iloc[:, 1:3].astype(int)
        return result
    else:
        return pd.concat([peak_data, result], axis=1)
