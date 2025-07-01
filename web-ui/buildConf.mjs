
import path from 'path'
import {cssModulesPlugin} from "@asn.aeb/esbuild-css-modules-plugin";

//import pkg from 'esbuild-plugin-external-global';
//const {externalGlobalPlugin} = pkg;




const aliasPlugin = {
  name: 'alias',
  setup(build) {
    build.onResolve({ filter: /^@web-gasket\// }, args => {
        const resolvedPath = path.join(
            '/home/maxl/dev/Marechal_pipelines/WebGasket',
            'js',
            args.path.replace('@web-gasket/', '')
        )
        return { path: resolvedPath }
    })
  }
}

export const conf = {
    entryPoints: ['src/App.jsx'],
    outfile: './build/bundle.js',
    //target: 'chrome58,firefox57,safari11',
    bundle: true,
    sourcemap: true,
    plugins: [
        /*
        externalGlobalPlugin({
            'react': 'window.React',
            'react-dom': 'window.ReactDOM'
        }),
         */
        aliasPlugin, cssModulesPlugin({
        // Optional. Will emit a `.css` bundle containing all of the imported css.
        emitCssBundle: {
            // Required. Will append `.css` at the end if missing
            filename: './build/bundle'
        }
    })]
}
