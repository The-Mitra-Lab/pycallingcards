from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_rankby = Optional[Literal["pvalues", "logfoldchanges"]]


def interactive(
    data: dict,
    chrom: str = "chr1",
    inter: int = 2,
    start: int = 0,
    end: int = 50000000,
    title: str = None,
):

    """\

    :param data:
        Annotated data matrix.
    :param chrom: Default is 'chr1'.
        The chromosome going to present.
    :param inter: Default is 2.
        The inter is the distance between each data.
    :param start: Default is 0.
        The start point of the plot.
    :param end: Default is 50000000.
        The end point of the plot.
    :param title: Default is None.
        The title of the plot.

    """

    add = 0

    fig = go.Figure()

    for i in range(len(data)):

        if data[i]["mode"] == "ccf":

            xvalue = np.array(
                data[i]["data"][
                    (data[i]["data"]["Chr"] == chrom)
                    & (data[i]["data"]["Start"] >= start)
                    & (data[i]["data"]["End"] <= end)
                ]["Start"]
            )

            yvalue = (
                np.log(
                    list(
                        data[i]["data"][
                            (data[i]["data"]["Chr"] == chrom)
                            & (data[i]["data"]["Start"] >= start)
                            & (data[i]["data"]["End"] <= end)
                        ]["Reads"]
                        + 1
                    )
                )
                - add
            )

            fig.add_trace(
                go.Scatter(
                    x=xvalue,
                    y=yvalue,
                    name=data[i]["name"],
                    mode="markers",
                    line=dict(color=data[i]["color"]),
                )
            )

            add += max(yvalue) + inter

        elif data[i]["mode"] == "peak":

            Start = list(
                data[i]["data"][
                    (data[i]["data"]["Chr"] == chrom)
                    & (data[i]["data"]["Start"] >= start)
                    & (data[i]["data"]["End"] <= end)
                ]["Start"]
            )
            End = list(
                data[i]["data"][
                    (data[i]["data"]["Chr"] == chrom)
                    & (data[i]["data"]["Start"] >= start)
                    & (data[i]["data"]["End"] <= end)
                ]["End"]
            )

            yvalue = []
            xvalue = []
            for peak in range(len(Start) - 1):
                yvalue.append(-add)
                yvalue.append(-add)
                yvalue.append(None)
                xvalue.append(Start[peak])
                xvalue.append(End[peak])
                xvalue.append(End[peak] + 1)

            fig.add_trace(
                go.Scatter(
                    x=xvalue,
                    y=yvalue,
                    name=data[i]["name"],
                    mode="lines+markers",
                    line=dict(color=data[i]["color"]),
                )
            )

            add += inter

        else:
            raise ValueError("Mode could only be among ['ccf','peak']")

    fig.update_layout(title_text=title)

    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    fig.update_yaxes(showticklabels=False)
    fig.show()
