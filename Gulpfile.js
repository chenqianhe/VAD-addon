const { series } = require('gulp');
const exec = require('child_process').exec;
const chalk = require('chalk');
const fs = require('fs');

const LOG = (content) => console.log(chalk.gray('[LOG] ' + content))
const INFO = (content) => console.info(chalk.green('[INFO] ' + content));
const ERROR = (content) => console.error(chalk.red('[ERROR] ' + content));
const NOTICE = (content) => console.info(chalk.blue('[NOTICE] ' + content));

const PLATFORM = process.platform;
INFO(`Platform: ${PLATFORM}`);

function clean(cb) {
    let cmd = null;
    INFO('Remove build dir')
    if (PLATFORM === 'darwin' || PLATFORM === 'linux') {
        cmd = 'rm -rf build';
    } else if (PLATFORM.startsWith('win')) {
        cmd = 'rd /S /Q build';
    }
    if (cmd) {
        fs.access('./build', fs.constants.F_OK, (err) => {
            if (!err) {
                exec(cmd, function (err, stdout, stderr) {
                    if (stdout) LOG(stdout);
                    if (stderr) LOG(stderr);
                    if (err) ERROR(stderr);
                    cb(err);
                });
            }
        })
    } else {
        ERROR('Your system is temporarily not supported.');
        cb(new ERROR('system is unsupported'));
    }
    cb();
}

const getCmakeParams = () => {
    const combineParams = (pre, cur) => PLATFORM.startsWith('win') ? `${pre} ${cur}` : `${pre} '${cur}'`;
    if (process.argv.length >= 2) {
        if (['clean', 'build', 'check'].includes(process.argv[2])) {
            return process.argv.length >= 3 ?
                process.argv.slice(3).reduce(combineParams, '') : '';
        } else {
            return process.argv.slice(2).reduce(combineParams, '');
        }
    }
    return '';
}

function build(cb) {
    const params = getCmakeParams()
    INFO('Build lib');
    if (params) INFO(`cmake-js params: ${params}`);
    const cmd = `npx cmake-js compile${params}`;
    INFO(cmd);
    exec(cmd, function (err, stdout, stderr) {
        if (stdout) LOG(stdout);
        if (stderr) LOG(stderr);
        if (err) ERROR(stderr);
        cb(err);
    });
}

function check(cb) {
    INFO('Check lib');
    fs.access('./build/Release', fs.constants.F_OK, (err) => {
        if (!err) {
            NOTICE('You can use ' +
                '\'const { vad } = require(path.join(__dirname, \'build/Release/vad_addon\'));\' ' +
                'to import the addon module.')
        } else {
            ERROR('There may be some errors here.')
            cb(new ERROR('node lib not exist'))
        }
    });
    cb();
}

exports.clean = clean;
exports.build = build;
exports.check = check;
exports.default = series(clean, build, check);
