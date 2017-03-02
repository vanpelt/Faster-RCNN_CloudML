LOCAL=0
CF=1
while getopts ":l:v" o; do
    case "${o}" in
	l) LOCAL=1;;
    v) CF=0;;
    esac
done
NAME=${@:$OPTIND:1}

if [ -z "$NAME" ]
then
	echo "You must specify a name.  If you want to run locally use -l, i.e. ./cloud-ml.sh -l my-name"
	exit 1
fi

JOB_NAME=${NAME/-/_}_${USER}_$(date +%Y%m%d_%H%M%S)

PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_BUCKET=gs://${PROJECT_ID}-ml
TRAIN_PATH=${TRAIN_BUCKET}/${JOB_NAME}

read -d '' TRAIN_ARGS_CF <<EOF
   --image_path $TRAIN_BUCKET/$NAME/source/*
   --label_path $TRAIN_BUCKET/$NAME/labels/*
   --label_type json
   --output_path $TRAIN_BUCKET/$NAME/train/
   --class_names_path $TRAIN_BUCKET/$NAME/class_names.json
   --network VGGnet_train
   --weights $TRAIN_BUCKET/VGG_imagenet.npy
   --cfg $TRAIN_BUCKET/$NAME/cfg.yml
   --gpu 0
   --iters 400
EOF

read -d '' TRAIN_ARGS_VOC <<EOF
   --imdb voc_2007_trainval
   --imdb_data_url $TRAIN_BUCKET/VOC2007
   --output_path $TRAIN_BUCKET/$NAME/train/
   --network VGGnet_train
   --weights $TRAIN_BUCKET/VGG_imagenet.npy
   --cfg $TRAIN_BUCKET/$NAME/cfg.yml
   --gpu 0
   --iters 10000
EOF

if [ $CF -eq 1 ]
then
    TRAIN_ARGS=$TRAIN_ARGS_CF
else
	TRAIN_ARGS=$TRAIN_ARGS_VOC
fi

FRCNN_PACKAGE=lib/dist/fast_rcnn-0.1.0.tar.gz
cd lib
python setup.py sdist
cd ../

if [ $LOCAL -eq 1 ] 
then
    echo "Running Locally -- ${TRAIN_ARGS}"
	pip install --upgrade --force-reinstall $FRCNN_PACKAGE
    gcloud beta ml local train \
       --package-path=tools \
       --module-name=tools.train_net \
       -- \
       ${TRAIN_ARGS}
else
    echo "Running in cloud -- ${TRAIN_ARGS}"
    gcloud beta ml jobs submit training ${JOB_NAME} \
	   --packages=${FRCNN_PACKAGE} \
	   --package-path=tools \
	   --module-name=tools.train_net \
	   --staging-bucket="${TRAIN_BUCKET}" \
	   --region=us-east1 \
	   --config=config/cloudml.yml \
	   -- \
	   ${TRAIN_ARGS}
fi
