# Marechal_pipelines


### SSh config for running on remote host gh1301.m 

```
Host gh1301
  Hostname gh1301.m
  user maxl
  ServerAliveInterval 600
  Compression yes
  IdentityFile ~/.ssh/id_rsa
  ProxyJump ip34.ccs.usherbrooke.ca
```

### Clone Repo

```
git clone --recurse-submodules --branch=main  git@github.com:jflucier/Marechal_pipelines.git

cd ./Marechal_pipelines
```

### Create a Virtual env :

```
python3.12 -m venv venv
. venv/bin/activate
numpy==2.2.3
pandas==2.2.3
cd ./DryPipe
pip install -e .
cd ..
```

### Setup an init script for your particular pipeline instance

Follow the example:   ./example/env-il2_inversefold.sh

It's a virtualenv init script for running pipelines instances from directory ./example

``` 
cd ./example

mkdir MyPipelineInstanceDir

# create samplesheet in ./MyPipelineInstanceDir/samplesheet.tsv

# init virtual env :

. ./env-il2_inversefold.sh `pwd`/MyPipelineInstanceDir

drypipe run

```
