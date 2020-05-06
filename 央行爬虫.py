import requests
from email.utils import formatdate




def center():
    #Tue, 28 Apr 2020 10:32:09
    dt = formatdate(None, usegmt=True)
    print(dt)
    headers={"User-Agent":"Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.90 Safari/537.36",
             "Referer":"http://www.pbc.gov.cn/",
             "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
             "Accept-Encoding":"gzip, deflate",
             "Accept-Language":"zh-CN,zh;q=0.9",
             "Cache-Control":"max-age=0",
             "Connection":"keep-alive",
             "Cookie":"wzws_cid=43f0d502c06f6cb1494cb27f37b51844a43b969229b4afb80ca895ca85f1092741b1c40fe7ae0f9f39cd1961d6766184215a0f197c611bdee91172e27d73f528089221f76931af5b1fb498057edee4137caa5ec76fbaf3f9f025e7bdae7e549eb70f28f3800e8e7821e32aea9e1eb03aaaa51e496b97795f83c43f535191537f",
             "Host":"www.pbc.gov.cn",
             "If-Modified-Since":dt,
             # "If-None-Match":'W/"10f97-20f26-5a45757ff1c40"',
             "Upgrade-Insecure-Requests":1

             }
    url="http://www.pbc.gov.cn"
    # print(type(stamp))
    resp=requests.get("http://www.pbc.gov.cn")
    print(resp.text)

if __name__=="__main__":
   center()