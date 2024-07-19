#!/bin/bash

echo "Enter Recording Name:"
read -r name

echo "Enter description:"
read -r desc

timestamp=$(date -Id)

touch data/00-Notes.md
echo -e "\n\n### $timestamp - $name\n" >>data/00-Notes.md
echo -e "$desc" >>data/00-Notes.md

docker compose run ros ros2 bag record --all --output="data/${timestamp}/${name}" --no-discovery --max-cache-size=1000

{
	echo -e '```'
	docker compose run ros ros2 bag info "data/${timestamp}/${name}"
	echo -e '```\n-----------\n\n'

} >>data/00-Notes.md
