# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:52:53 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""

#pylint: disable=C0326
import logging

import click
from download import do_download
from train_test import run_train, run_test, run_infer


LOG = logging.getLogger( __name__ )
LOG.setLevel( 1 )
log = LOG.log #pylint: disable=C0103
#%%
DATA_URL = "http://benchmark.ini.rub.de/Dataset_GTSDB/GTSDB_CppCode.zip"
#DATA_URL="http://goldseek.com/news/GoldSeek/2007/6-4mb/7.JPG"
DOWNLOAD_DEST_DIR = "images/"
#%%

@click.group()
def cli():
    """Just a command group"""
    #import sys
    #print( sys.version )
    pass


@click.command()
@click.option( '--data_url', default=DATA_URL,
               help='url of a zip file to download and decompress' )
@click.option( '--dest_dir', default = DOWNLOAD_DEST_DIR,
               help='destination dir for decompressed contents' )
def download( data_url, skip_download=False, remove_zip_after=True ) :
    """Download a zipfile from DATA_URL decompress it and put results in dest_dir """
    do_download( data_url,
                 dest_dir='tmp_dir/',
                 skip_download=skip_download,
                 remove_zip_after=remove_zip_after )

#%%
@click.command()
@click.option( '-m', help='name of model to train')
@click.option( '-d', help='directory holding training images')

@click.option( '-v', default=0, help='verbosity level')
@click.option( '--vid', default=None )
#pylint: disable=C0103
def train( m, d, v, vid ) :
    """Run train subcommand for model_name=m and images_dir=d"""
    assert m, "Need to pass a model name with -m"
    assert d, "Need to pass a dir name with -d"

    model_name = m
    images_dir = d
    logl = 32  if v > 0 else 0

    log( logl, "Training model=%s on images of training dir=%s",
         model_name, images_dir )

    run_train( model_name=m, images_dir=d, valid_images_dir=vid, logl=logl )


@click.command()
@click.option( '-m', help='name of model to test')
@click.option( '-d', help='directory holding test images')
@click.option( '-v', default=0, help='verbosity level')
#pylint: disable=C0103
def test( m, d, v ) :
    """Run test subcommand for model_name=m and images_dir=d"""
    assert m, "Need to pass a model name with -m"
    assert d, "Need to pass a dir name with -d"

    model_name = m
    images_dir = d
    logl = 32 if v > 0 else 0

    log( logl, "Testing model=%s on images of training dir=%s",
         model_name, images_dir )

    run_test( model_name=m, images_dir=d, logl=logl )


@click.command()
@click.option( '-m', help='name of model')
@click.option( '-d', help='directory holding  images')
@click.option( '-v', default=0, help='verbosity level')
#pylint: disable=C0103
def infer( m, d, v ) :
    """Run infer subcommand for model_name=m and images_dir=d"""
    assert m, "Need to pass a model name with -m"
    assert d, "Need to pass a dir name with -d"

    model_name = m
    images_dir = d
    logl = 32 if v > 0 else 0

    log( logl, "Infering with model=%s on images of training dir=%s",
         model_name, images_dir )

    run_infer( model_name=m, images_dir=d, logl=logl )

cli.add_command( download )
cli.add_command( train )
cli.add_command( test )
cli.add_command( infer )

#%%

if __name__ == '__main__' :
    cli()
