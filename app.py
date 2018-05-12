# -*- coding: utf-8 -*-
"""
Created on Fri May 11 16:52:53 2018

@author: mrestrepo
@company: Yuxi Global (www.yuxiglobal.com)
"""

#pylint: disable=C0326

import click
from download import do_download
from train import run_train, run_test

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
#pylint: disable=C0103
def train( m, d, v ) :
    """Run train subcommand for model_name=m and images_dir=d"""
    assert m, "Need to pass a model name with -m"
    assert d, "Need to pass a dir name with -d"

    model_name = m
    images_dir = d

    if v > 0 :
        print( "running training for model=%s on images of training dir=%s" %
               (model_name, images_dir)  )

    run_train( model_name=m, images_dir=d, verbose=v  )


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
    if v > 0 :
        print( "running test of model=%s on images of training dir=%s" %
               (model_name, images_dir) )

    run_test( model_name=m, images_dir=d, verbose=v )


cli.add_command( download )
cli.add_command( train )
cli.add_command( test )


#%%

if __name__ == '__main__' :
    cli()
