import React from 'react'

import {createApp} from "@web-gasket/App.jsx"
import DPFoldArgsEditor from "./DPFoldArgsEditor";

const RenderPipelineSettingsEditor = ({pipelineInstance, pipelineArgsDispatcher, renderGeneric}) => {

    switch (pipelineInstance.type) {
        case "dp-fold": {

            return <DPFoldArgsEditor
                pipelineInstance={pipelineInstance}
                pipelineArgsDispatcher={pipelineArgsDispatcher}
            />
        }
        default:
            return renderGeneric()
    }
}

createApp(RenderPipelineSettingsEditor)