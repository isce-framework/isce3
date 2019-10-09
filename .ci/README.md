# NISAR dockerImages

Dockerfile recipes for NISAR

## Docker tutorials
------
1. https://medium.com/@deepakshakya/beginners-guide-to-use-docker-build-run-push-and-pull-4a132c094d75


## Getting started on OS X
-------
1. sudo port install docker docker-machine
    - Alternately download docker for mac

2. sudo port install virtualbox
    - Alternately install virtualbox for mac

### First time setup
--------
1. We need to create a docker machine that will control all your docker instances. Lets name this machine "dev".

```bash
> docker-machine create dev --driver virtualbox
```

This will take a while to run and will build a small VM. Once, you have this - you are ready to start. You can verify that your "dev" instance is running with 

```bash
> docker-machine ls
```

If the machine is not running, start it with

```bash
> docker-machine start dev
```


### Every time you login/ restart
-------
Every time you login / restart, you will need set appropriate environment variables for docker to work. To use the "dev" docker-machine

```bash
> docker-machine start dev
> eval "$(docker-machine env dev)"
```

Add this to your modules or a startup script for docker. 

You can test that docker is properly setup now by testing 

```bash
docker run hello-world
```

If you don't have the hello-world image, this will pull a 2kB docker image and run it for you.


## How to build an image from the recipes
-----------

The following command will build a container tagged "centosbase:v0" from the recipe in the centosbase folder.

```bash
> docker build -t centosbase:v0 images/centosbase
```

Confirm that this got created with
```bash
>docker image ls 
```

## How to connect to a container
-------------

The following command will create an instance of your container and allow you to login

```bash
> docker run -it centosbase:v0 bash
```

After closing the shell, dont forget to clean up the instance that was created following these 2 steps:

1. Get instance id

```bash
> docker ps -a
```

2. Remove the instance 

```bash
> docker rm idfromprevcmd
```

## Shutting down docker
------------

To shut down the VM running docker 

```bash
> docker-machine stop dev
```

It is recommended to add the docker-machine start / stop commands as aliases to a module file and run these after module load and before module unload.
