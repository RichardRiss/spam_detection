#year="2003"
years=(2014 2015 2016 2017 2018 2019 2020 2021 2022)


shopt -s nullglob
for year in "${years[@]}"; do
    echo "Working on ${year}"
    csv="${year}.csv"
    rm -f $csv
    echo "email,label" > $csv

    for filename in SpamArchive/$year/*.lorien SpamArchive/$year/*.txt; do
        #echo $filename >> $csv
        #(echo -n "\"b\'" ; tr '\r\n' ' ' < $filename ; echo "\",1") >> $csv
        #l=$(sed -e ':a;N;$!ba;s/\n/,/g' -e 's/\"//g' < $filename)
        l=$(awk -v RS= 'NR>1' $filename)
        #echo $l
        (echo -n "\"b\'" ; tr '\r\n' ' ' <<< $(sed -e ':a;N;$!ba;s/\n/,/g' -e 's/\"//g' <<< $l) ; echo "\", 0.0") >> $csv
    done
done