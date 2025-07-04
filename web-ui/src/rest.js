

export const getDPFoldCustomFilesStatus = pid =>
    fetch(`/api/dpFoldFilesStatus${pid}`)
