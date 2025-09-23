

import esbuild from 'esbuild';
import {conf} from "./buildConf.mjs";


esbuild.build({
    ...conf,
    outfile: '',
    entryNames: '[dir]/bundle-[hash]',
    outdir: './build',
    minify: true
})