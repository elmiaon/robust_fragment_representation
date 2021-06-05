CORPUS=('JW300' 'QED' 'TED2020')
SRC_LANGS=('fr' 'de' 'th' 'ar')
TARGET_LANGS=('en')

if [[ -d "/mount/data" ]]; then
    echo "data exists"
else
    mkdir "/mount/data"
fi

if [[ -d "/mount/data/raw" ]]; then
    echo "data/raw exists"
else
    mkdir "/mount/data/raw"
fi

if [[ -d "/mount/data/raw/opus" ]]; then
    echo "data/raw/opus exists"
else
    mkdir "/mount/data/raw/opus"
fi

for i in "${CORPUS[@]}"; do
    if [[ -d "/mount/data/raw/opus/$i" ]]; then
        echo "/mount/data/raw/opus/$i"
    else
        echo "create /mount/data/raw/opus/$i"
        mkdir "/mount/data/raw/opus/$i/"
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