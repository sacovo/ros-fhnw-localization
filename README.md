# ROS FHNW Project-Template

This template should be copied to create new software for the fhnw rover.

Only create a new repository after consideration with the whole team, as the number of repositories shouldn't be to high, in order to keep the project easy to manage.

Every subsystem should be covered in its own repository, only deviate from this for very good reasons, after discussion and make sure to document the reasons.

The repository fhnw_interfaces contains ros message definitions, services and actions used to communicate between the different subsystems. Try to use already existing message definitions from std_msgs, geometry_mgs, nav_msgs and other builtin packages.

## Get started

After cloning the repository you can use the provided `docker-compose.yml` file together with VS Code's dev-containers to get a working ros instance.

To create a new ros package use the following command:

```
ros2 pkg create --build-type ament_cmake  --maintainer-email your@email --node-name node_name pkg_name
ros2 pkg create --build-type ament_python --maintainer-email your@email --node-name node_name pkg_name
```

### Dependencies

To add packages you can use `docker/apt.txt` and `docker/apt-dev.txt` files, graphical tools only needed for developement should be put in apt-dev.txt to keep the image size small. Build dependencies also belong into the apt-dev.txt file, only add the bare minimum into apt.txt.

Python dependencies are managed using pip-compile (included in the image for the dev-container). Put new packages into requirements.in and run pip-compile while in the docker folder. This will write the exact dependencies into requirements.txt which is used to build the images.


### Documentation

After you are setup, make sure to replace this README.md with an actual description of your project. The description should include the purpose of the project as well as techincal details about the solution. Information that is the same over all projects (like how to use the docker images, or how to deploy the tools or general information about ros) does not belong inside the project documentation.

Add a section about the nodes, services and topics that the project provides to the outide. Not every node, topic or service needs to be documented, only the ones that are of interested to external nodes.

## Code style

For python use black (included as extension in dev-container) and for C++ use the provided formatter as well.

## Deployment

See ...

## Example Documentation

This project implements the ... for the FHNW Rover.

## Hardware needed

Does the project need certain hardware to run, access to ports, ...

## Nodes

## Topics

## Services

## Actions

## Notes
