import os
import json
import time
import requests
import numpy as np
import pandas as pd
import random


from typing import  Optional, Literal
_reference2 = Optional[Literal["hg38","mm10","sacCer3"]]

def qbed_to_bedgraph(expdata,number = 4):
    
    chrm = list(expdata["Chr"].unique())

    chrnew = []
    startnew = []
    endnew = []
    count = []
    number = 4

    for chrom in chrm:
        curChrom = list(expdata[expdata["Chr"] == chrom]["Start"]) 
        start,counts = np.unique(curChrom, return_counts=True)

        chrnew = chrnew + [chrom] * len(start)
        startnew = startnew + (list(start))
        endnew = endnew + list(start+number)
        count = count + list(counts)

    return pd.DataFrame(list(zip(chrnew, startnew, endnew, count)))


class BrowserHelperClient(object):
    _base_url_protocol = "https"
    _base_url_host = "10.20.127.22"
    _base_url_port = 10981
    _base_url = ""

    def __init__(self):
        self._base_url = self.make_url()

    def update_url(self):
        self._base_url = self.make_url()

    def make_url(self):
        port = ":%s" % self._base_url_port
        if self._base_url_port in ["22", 22]:
            port = ""
        url = "{protocol}://{host}{port}".format(
            protocol=self._base_url_protocol, host=self._base_url_host, port=port
        )
        return url

    def set_base_url_protocol(self, x):
        self._base_url_protocol = x
        self.update_url()

    def set_base_url_host(self, x):
        self._base_url_host = x
        self.update_url()

    def set_base_url_port(self, x):
        self._base_url_port = x
        self.update_url()

    def submit(self, input_files, assembly="hg38"):
        files = []
        for file_name, file_path in input_files.items():
            files.append(("file", (file_name, open(file_path).read())))

        surl = self._base_url + "/file_upload"

        response = requests.post(surl, files=files, data={"assembly": assembly}, verify=False)

        assert response.status_code == 200
        task_id = response.text
        return task_id

    def retrieve_simple(self, task_id):
        surl = self._base_url + "/retrieve?list_id=" + task_id
        response = requests.get(surl, verify=False)
        res = json.loads(response.text)[0]
        # print(res)
        return res

    def retrieve(self, task_id):
        while True:
            res = self.retrieve_simple(task_id)
            if res["finished"]:
                break
            time.sleep(1)
        return res

    def gburl(self, server_result):
        return server_result["result"]["gburl"]

    def simple_request(self, input_files, assembly="hg38"):

        task_id = self.submit(input_files, assembly=assembly)
        result_json = self.retrieve(task_id)
        gburl = self.gburl(result_json)
        return gburl


def WashU_browser_url(
    qbed: dict = {},
    bed: dict = {},
    genome: _reference2 = 'hg38',
    dirc: str = 'WashU_cashe',
    remove: bool =  False
    ):
    
    """\
    Display qbed/ccf, bed data in `WashU Epigenome Browser <http://epigenomegateway.wustl.edu/browser/>`
    :param qbed: Default is blank.
        A dictionary for input qbed/ccf data key as the name to display and value is the path or data of the file. Prefer path.
    :param bed: Default is blank.
        A dictionary for input bed data(peak data) key as the name to display and value is the path or data of the file. Prefer path.
    :param genome: Default is 'mm10'.
        Genome to display.
        Currently, `'hg38'`, `'mm10'`, `'sacCer3'` are provided only.
    :param dirc:  Default is 'WashU_cashe'.
        The dirctory for all the cache files.
    :param remove: Default is `False`.
        Weather to remove the dirc or not.
        
    """

    bhc = BrowserHelperClient()

    # TODO Remember to set it to False
    if False:
        bhc.set_base_url_host("localhost")
        bhc.set_base_url_port("10981")
        bhc.set_base_url_protocol("https")

    burl = bhc.make_url()
    print(burl)

    file = {}

    if genome == "hg38" or genome == "mm10":
        number = 4
    elif genome == "sacCer3" :
        number = 1
    else:
        raise ValueError('Please input correct genome')

    if not os.path.exists(dirc):
        os.system('mkdir '+ dirc)

    qbed_keys = list(qbed.keys())
    for key in qbed_keys:
        if type(qbed[key]) == str:

            file[key] = qbed[key]

            expdata = pd.read_csv(qbed[key], sep = "\t",  header = None, names =  ["Chr", "Start", "End", "Reads", "Direction", "Barcodes"])

            expdatagraph = qbed_to_bedgraph(expdata,number = number)
            randomnum = str(random.random())
            expdatagraph.to_csv(dirc+"/"+randomnum+".bedgraph",sep ="\t",header = None, index = None)
            file[key+'density'] = dirc+"/"+randomnum+".bedgraph"

        elif type(qbed[key]) == pd.DataFrame:

            randomnum = str(random.random())
            qbed[key].to_csv(dirc+"/"+randomnum+".qbed",sep ="\t",header = None, index = None)
            file[key] = dirc+"/"+randomnum+".qbed"

            expdatagraph = qbed_to_bedgraph(qbed[key],number = number)
            randomnum = str(random.random())
            expdatagraph.to_csv(dirc+"/"+randomnum+".bedgraph",sep ="\t",header = None, index = None)
            file[key+'density'] = dirc+"/"+randomnum+".bedgraph"

        else:
            raise ValueError('Please input correct form of ' +key + " in qbed")
            
    print("All qbed addressed")

    bed_keys = list(bed.keys())

    for key in bed_keys:

        if type(bed[key]) == str:

            file[key] = bed[key]

        elif type(bed[key]) == pd.DataFrame:

            randomnum = str(random.random())
            bed[key].to_csv(dirc+"/"+randomnum+".bed",sep ="\t",header = None, index = None)
            file[key] = dirc+"/"+randomnum+".bed"

        else:
            raise ValueError('Please input correct form of ' +key + " in bed")
            
    print("All bed addressed")
    print("Uploading files")


    gburl = bhc.simple_request(file, assembly=genome)
    print(gburl)
    
    if remove:
        os.system('rm -r '+dirc)