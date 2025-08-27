
import path from 'path'
import {cssModulesPlugin} from "@asn.aeb/esbuild-css-modules-plugin"

function removeLastDir(pathStr) {
    // Remove trailing slash if present
    pathStr = pathStr.replace(/\/$/, '');
    // Split and remove last part
    const parts = pathStr.split('/');
    parts.pop();
    return parts.join('/') || '/';
}


const aliasPlugin = {
  name: 'alias',
  setup(build) {
    build.onResolve({ filter: /^@web-gasket\// }, args => {

        const scriptUrl = new URL(import.meta.url);
        const scriptDir = removeLastDir(scriptUrl.pathname.replace(/\/[^\/]+$/, ''))

        const resolvedPath = path.join(
            scriptDir, 'WebGasket', 'js', args.path.replace('@web-gasket/', '')
        )

        //console.log(resolvedPath)
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
        aliasPlugin, cssModulesPlugin({
        // Optional. Will emit a `.css` bundle containing all of the imported css.
        emitCssBundle: {
            // Required. Will append `.css` at the end if missing
            filename: './build/bundle'
        }
    })]
}
