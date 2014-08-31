sudo yum -y install htop python-pip tmux
pip install -r requirements.txt --user
cd ..
python setup.py install --user
cd examples
~/spark-ec2/copy-dir ~/.local
