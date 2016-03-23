#!/bin/bash
python csetup.py build
cp build/lib.linux-x86_64-2.7/realmd/_ac.so ./realmd/
cp build/lib.linux-x86_64-2.7/mdfc/_mdfc.so ./mdfc/
