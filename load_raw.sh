CORPUS=('JW300' 'QED' 'TED2020')
SRC_LANGS=('fr' 'de' 'th' 'ar')
TARGET_LANGS=('en')

if [[ -d "/mount/corpus" ]]; then
    echo "corpus exists"
else
    mkdir "/mount/corpus"
fi

if [[ -d "/mount/corpus/raw" ]]; then
    echo "corpus/raw exists"
else
    mkdir "/mount/corpus/raw"
fi

if [[ -d "/mount/corpus/raw/opus" ]]; then
    echo "corpus/raw/opus exists"
else
    mkdir "/mount/corpus/raw/opus"
fi

for i in "${CORPUS[@]}"; do
    if [[ -d "/mount/corpus/raw/opus/$i" ]]; then
        echo "/mount/corpus/raw/opus/$i"
    else
        echo "create /mount/corpus/raw/opus/$i"
        mkdir "/mount/corpus/raw/opus/$i/"
    fi

    for j in "${SRC_LANGS[@]}"; do
        for k in "${TARGET_LANGS[@]}"; do
            if [[ ! -f "corpus/raw/opus/$i/$j-$k.$j" && ! -f "corpus/raw/opus/$i/$j-$k.$k" ]]; then
                # echo "${j}_${k} Not found"
                opus_read -q -d $i -s $j -t $k -wm moses -w corpus/raw/opus/$i/$j-$k.$j corpus/raw/opus/$i/$j-$k.$k
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