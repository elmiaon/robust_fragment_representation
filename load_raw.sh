CORPUS=('JW300' 'QED' 'TED2020')
SRC_LANGS=('fr' 'de' 'th' 'ar')
TARGET_LANGS=('en')

if [[ -d "data" ]]; then
    echo "data exists"
else
    mkdir "data"
fi

if [[ -d "data/raw" ]]; then
    echo "raw exists"
else
    mkdir "data/raw"
fi

if [[ -d "data/raw/opus" ]]; then
    echo "raw/opus exists"
else
    mkdir "data/raw/opus"
fi

for i in "${CORPUS[@]}"; do
    if [[ -d "data/raw/opus/$i" ]]; then
        echo "data/raw/opus/$i"
    else
        echo "create data/raw/opus/$i"
        mkdir "data/raw/opus/$i/"
    fi

    for j in "${SRC_LANGS[@]}"; do
        for k in "${TARGET_LANGS[@]}"; do
            if [[ ! -f "data/raw/opus/$i/$j-$k.$j" && ! -f "data/raw/opus/$i/$j-$k.$k" ]]; then
                # echo "${j}_${k} Not found"
                opus_read -q -d $i -s $j -t $k -wm moses -w data/raw/opus/$i/$j-$k.$j data/raw/opus/$i/$j-$k.$k
            else
                echo "$i $j-$k Exists"
            fi
        done
    done

    for j in "${SRC_LANGS[@]}"; do
        rm ${i}_latest_xml_${j}.zip
    done

    for k in "${TARGET_LANGS[@]}"; do
        rm ${i}_latest_xml_${k}.zip
    done

    for j in "${SRC_LANGS[@]}"; do
        for k in "${TARGET_LANGS[@]}"; do
            rm ${i}_latest_xml_${j}-${k}.xml.gz
            rm ${i}_latest_xml_${k}-${j}.xml.gz
        done
    done
done