Training Word2Vec with a small Wiki Dump:

```shell
sudo yum install python-pip
pip install -r requirements.txt

curl http://cs.fit.edu/~mmahoney/compression/enwik8.zip -o enwik8.zip
unzip enwik8.zip
./wikifilter.pl enwik8 >enwiki
~/ephemeral-hdfs/bin/hadoop fs -cp file:///root/deepdist/examples/enwiki /enwiki

curl http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 -o enwiki-latest-pages-articles.xml.bz2
```
