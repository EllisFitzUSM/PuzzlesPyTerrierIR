
- Requirments
  -    
  
        pip install -r path/to/requirements.txt

- How to run
  - 
  - If *pt_index* is not present on your local machine, you will need to do the index

        python __main__.py -dc path/to/doc.json
  
  - Then feel free to provide it with a qrel and a topics file
  
        python __main__.py -qrels path/to/qrel.tsv path/to/topics.json

  - More can be done in addition, just use the help command

        python __main__.py -h
