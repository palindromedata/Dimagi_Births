#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:25:05 2018

@author: ksharpey
"""

Hi Neal

Here's a general update, but I don't think anything contravertial or spectacular.

I'm currently working on:
    - Extra EF as discussed
    - risk probability distribution - I wondered if even if there's little uplift from the baseline, if the old DataProphet Graph were still interesting in knowing who is high risk? However I wondered if we weren't too interested in the risk scores unti the accuracy is higher.
      All the same, just trying to make it easy for the script to produce interesting outputs on that (examples copied below, but which have been hand manipulated for the 1 pager). 
    - trying to automate that it creates a better "mirror" target for the Majority target. E.g. "Births at home including blanks" would be the mirror of "births in hospital"
    - Automating assesment the "optimal" n features needed for a target. I hand wrote a bunch of this, only to find yesterday that there's a library to do that for you aswell! Good times. (https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization)
        The hand written initial results interestingly 
    - Automating the assement of n RTP used for training vs peformance. I did some rough tests by hand, but the framework change to automate that is more complicated as I should really be breaking up my long code now into libraries. Coming up against my python limitaitons - hope to get Cory/Drew advice next week)
    - I did install wrangler, and it handily counts the blanks, calculates distributionas for each column. I haven't dug much further with it yet, but def worth while and should save some time on the next data set