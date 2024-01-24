#!/bin/bash

function_transfer () {
	for f in ${SOURFOLDER}/*.jpg; do
		FILENAME="$(echo "${f%.jpg}" | cut -d"/" -f2)"	
		mv -- "${SOURFOLDER}/${FILENAME}.jpg" "${DESTFOLDER}/${FILENAME}-${NUM}.jpg"
	done

	for f in ${SOURFOLDER}/*.jpeg; do
		FILENAME="$(echo "${f%.jpeg}" | cut -d"/" -f2)"
		mv -- "${SOURFOLDER}/${FILENAME}.jpeg" "${DESTFOLDER}/${FILENAME}-${NUM}.jpeg"
	done
}

SOURFOLDER="rosemary-herb(copy)"
DESTFOLDER="rosemary-herb"
NUM="1"
function_transfer

SOURFOLDER="rosemary-herb(anothercopy)"
DESTFOLDER="rosemary-herb"
NUM="2"
function_transfer

SOURFOLDER="rosemary-herb(3rdcopy)"
DESTFOLDER="rosemary-herb"
NUM="3"
function_transfer

#-------------------------------------------------

SOURFOLDER="sage-herb(copy)"
DESTFOLDER="sage-herb"
NUM="1"
function_transfer

SOURFOLDER="sage-herb(anothercopy)"
DESTFOLDER="sage-herb"
NUM="2"
function_transfer

SOURFOLDER="sage-herb(3rdcopy)"
DESTFOLDER="sage-herb"
NUM="3"
function_transfer

#-------------------------------------------------

SOURFOLDER="thyme-herb(copy)"
DESTFOLDER="sage-herb"
NUM="1"
function_transfer

SOURFOLDER="thyme-herb(anothercopy)"
DESTFOLDER="thyme-herb"
NUM="2"
function_transfer

SOURFOLDER="thyme-herb(3rdcopy)"
DESTFOLDER="thyme-herb"
NUM="3"
function_transfer



