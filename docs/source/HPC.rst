=======================
Multiprocessing and HPC
=======================

Currently `symmer <https://github.com/UCL-CCS/symmer>`_ uses `ray <https://github.com/ray-project/ray>`_ by default to accelerate
the codebase by distributing problems over multiple cores if available. For standard use this works seamlessly and the user
doesn't need to do anything. However, when deploying on HPC systems a couple of things
need to be set. For further information see `link <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_.
Below we comment on how to implement these tools on a SLURM system when ray is used. However, first we show an easier fix that will work on all linux based systems,
but can have issues with windows OS.

+++++
Easy fix
+++++
Note these fixes can be implemented for local use on a laptop/desktop too. When importing symmer the following code
should be run before any other symmer functionality is used:

.. code-block:: bash

    from symmer import process
    process.method = 'mp' # for multiprocessing to be used instead of ray

An alternate fix is to turn multiprocessing off via:

.. code-block:: bash

    from symmer import process
    process.method = 'single_thread'  # stops all multiprocessing

Finally, ray can be turned back by doing

.. code-block:: bash

    from symmer import process
    process.method = 'ray'  # makes ray the multiprocessing library (set to this by default)

+++++
Ray and SLURM
+++++
Note these steps are **NOT** required for local use on a laptop/desktop.

In the sbatch script for a job the user needs to allocate nodes specifically for Ray, which will then allocate
resources appropriately.

.. code-block:: bash

    #SBATCH --tasks-per-node=1
    #SBATCH --exclusive

A user can then set further resources:

.. code-block:: bash

    ### Modify this according to your Ray workload.
    #SBATCH --cpus-per-task=5
    #SBATCH --mem-per-cpu=1GB
    ### Similarly, you can also specify the number of GPUs per node.
    ### Modify this according to your Ray workload. Sometimes this
    ### should be 'gres' instead.
    #SBATCH --gpus-per-task=1

This will guarantee that each Ray worker will have access to the required resources, here
5 CPUs and 5GB memory per node.

Finally, in order for Ray to work properly we need to set the Ray head node. For further details
see `link <https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html>`_. For most
purposes all that is required is to copy the following code into the bash script, prior to running
the python script:

.. code-block:: bash

    ## setup ray
    head_node=$(hostname)
    head_node_ip=$(hostname --ip-address)
    # if we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
      head_node_ip=${ADDR[1]}
    else
      head_node_ip=${ADDR[0]}
    fi
    fi
    port=6379

    echo "STARTING HEAD at $head_node"
    echo "Head node IP: $head_node_ip"
    srun --nodes=1 --ntasks=1 -w $head_node start-head.sh $head_node_ip &
    sleep 10

    worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
    srun -n $worker_num --nodes=$worker_num --ntasks-per-node=1 --exclude $head_node start-worker.sh $head_node_ip:$port &
    sleep 5
    ## finished ray setup

    ### submit python script
    ## e.g. python test.py "$SLURM_CPUS_PER_TASK"
