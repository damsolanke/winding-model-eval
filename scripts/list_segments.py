"""List PHerc0139 segment mesh files (tifxyz) and their sizes from the open-data bucket."""
import json
import re
import sys
import urllib.request

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


def list_keys(prefix, delimiter=None):
    keys, prefixes, token = [], [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}"
        if delimiter:
            url += f"&delimiter={delimiter}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        text = urllib.request.urlopen(url).read().decode()
        keys += list(zip(re.findall(r"<Key>([^<]*)</Key>", text),
                         map(int, re.findall(r"<Size>([^<]*)</Size>", text))))
        prefixes += re.findall(r"<Prefix>([^<]*)</Prefix>", text)[1:]
        m = re.search(r"<NextContinuationToken>([^<]*)<", text)
        if not m:
            return keys, prefixes
        token = m.group(1)


import urllib.parse

if __name__ == "__main__":
    sample = sys.argv[1] if len(sys.argv) > 1 else "PHerc0139"
    _, segs = list_keys(f"{sample}/segments/", "/")
    out = {}
    for seg in segs:
        keys, _ = list_keys(seg + "mesh/")
        out[seg] = [(k[len(seg):], s) for k, s in keys if k.endswith((".tif", "meta.json"))]
    json.dump(out, sys.stdout, indent=1)
