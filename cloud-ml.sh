LOCAL=0
INSTALL=0
while getopts ":l" o; do
    case "${o}" in
	l) LOCAL=1;;
	i) INSTALL=1;;
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

read -d '' TRAIN_ARGS <<EOF
   --image_path $TRAIN_BUCKET/$NAME/source/*
   --label_path $TRAIN_BUCKET/$NAME/labels/*
   --label_type json
   --output_path $TRAIN_BUCKET/$NAME/train/
   --class_names_path $TRAIN_BUCKET/$NAME/class_names.json
   --network VGGnet_train
   --weights $TRAIN_BUCKET/VGG_imagenet.npy
   --cfg $TRAIN_BUCKET/$NAME/cfg.yml
   --gpu 0
   --iters 1000
EOF

FRCNN_PACKAGE=lib/dist/fast_rcnn-0.1.0.tar.gz
cd lib
python setup.py sdist
cd ../

if [ $LOCAL -eq 1 ] 
then
    echo "Running Locally -- ${TRAIN_ARGS}"
	if [ $INSTALL -eq 1 ]
	then
		pip install --upgrade --force-reinstall $FRCNN_PACKAGE
	fi
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
