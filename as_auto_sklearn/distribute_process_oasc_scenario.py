#! /usr/bin/env python3

import argparse
import os

import misc.automl_utils as automl_utils
import misc.shell_utils as shell_utils
import misc.ssh_utils as ssh_utils
import misc.utils as utils

import as_auto_sklearn.as_asl_command_line_utils as clu

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Distribute runs of process-oasc-scenario around a "
        "cluster using password-less ssh")

    clu.add_config(parser)
    parser.add_argument('oasc_scenarios_dir')

    clu.add_num_cpus(parser)
    clu.add_cv_options(parser)
    clu.add_scheduler_options(parser)

    automl_utils.add_automl_options(parser)
    automl_utils.add_blas_options(parser)


    parser.add_argument('--dry-run', help="If this flag is given, then the "
        "commands will be printed to the screen, but not executed", 
        action='store_true')

    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    # make sure the config file exists
    if not os.path.exists(args.config):
        msg = "Could not find the config file: {}".format(args.config)
        raise FileNotFoundError(msg)

    config = clu.get_config_options_string(args)
    cpus_str = clu.get_num_cpus_options_string(args)
    cv_str = clu.get_cv_options_string(args)
    scheduler_str = clu.get_scheduler_options_string(args)
    automl_str = automl_utils.get_automl_options_string(args)
    blas_str = automl_utils.get_blas_options_string(args)
    logging_str = logging_utils.get_logging_options_string(args)

    training_dir = os.path.join(args.oasc_scenarios_dir, "train")
    testing_dir = os.path.join(args.oasc_scenarios_dir, "test")
    scenarios = utils.listdir_full(training_dir)
    scenarios = [
        utils.get_basename(s) for s in scenarios if os.path.isdir(s)
    ]

    commands = []
    for scenario in scenarios:
        train = os.path.join(training_dir, scenario)
        test = os.path.join(testing_dir, scenario)
        cmd = [
            "process-oasc-scenario",
            args.config,
            train,
            test,
            cpus_str,
            cv_str,
            scheduler_str,
            automl_str,
            blas_str,
            logging_str
        ]

        cmd = ' '.join(cmd)
        commands.append(cmd)
            
    # if this is a dry run, just print the commands and quit
    if args.dry_run:
        for cmd in commands:
            msg = "Skipping due to --dry-run flag"
            logger.info(cmd)
            logger.info(msg)

        return

    # otherwise, make the remote calls
    node_list, proc_list = ssh_utils.distribute_all(commands, args.node_list, 
        args.connection_timeout, args.max_tries)

    ret = ssh_utils.wait_for_all_results(commands, node_list, proc_list)

    if ret is not None:
        (return_codes, stdouts, stderrs) = ret

        if args.store_out is not None:
            with open(args.store_out, 'w') as out:
                for i in range(len(return_codes)):
                    out.write(commands[i])
                    out.write("\n")
                    out.write(node_list[i])
                    out.write("\n")
                    out.write("return code: {}".format(return_codes[i]))
                    out.write("\n")
                    out.write("stdout: {}".format(stdouts[i]))
                    out.write("\n")
                    out.write("stderr: {}".format(stderrs[i]))
                    out.write("\n")

if __name__ == '__main__':
    main()
