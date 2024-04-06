diff $1 $2 
if [ "$?" == "0" ]
then
	echo "NO Difference Found"
else
	echo "ERROR - Difference in output"
fi
