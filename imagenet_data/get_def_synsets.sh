while read p; do
	if [[ ! $p = "" ]]; then
		echo $p":"
	fi
	if [[ $p = "" ]]
	then
		echo "#####################"
	elif [[ $p = "no_strata" ]]
	then
		echo "NO STRATA"
	else
		./ImageNet_SIFT/get_meaning_synset.sh $p
		echo "------"
	fi
done < strata_sorted.txt
