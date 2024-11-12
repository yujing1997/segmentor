### Downloading Pan-Cancer-Nuclei-Seg 

- From the IBM Aspera server, download the Pan-Cancer-Nuclei-Seg dataset from the following link: [Pan-Cancer-Nuclei-Seg](https://ibm.box.com/s/7v1v7k7z1v7k7z1v7k7z1v7k7z1v7k7z)

- Huizhong's attempt: 

    - `module load ruby export PATH=$PATH:$HOME/.gem/gems/aspera-cli-4.19.0/bin/`
        - works! 

    - `gem install rainbow --user-install`
    - `gem install terminal-table --user-install`
        
    - `ascli faspex5 packages receive \--url='....putYourLinkHere...'`
        - Did not work! 
        - Attempted: `ascli faspex5 packages receive \--url='https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6Ijc2NiIsInBhc3Njb2RlIjoiMDgwNjQ5ZDRmOWRjZjMwMzllMDMyN2Y5Njk2MTU5NTkxNWY4MjNmMiIsInBhY2thZ2VfaWQiOiI3NjYiLCJlbWFpbCI6ImhlbHBAY2FuY2VyaW1hZ2luZ2FyY2hpdmUubmV0In0=&redirected=true'`


- /home/yujingz/scratch/NUCLEI_SIZE_CODE/HZ/PKG - Pan-Cancer-Nuclei-Seg/metadata-files/cesc_meta/TCGA-2W-A8YY-01Z-00-DX1.2BEC2531-DA98-429B-83BB-3428D3B6FB1E.svs.tar.gz

