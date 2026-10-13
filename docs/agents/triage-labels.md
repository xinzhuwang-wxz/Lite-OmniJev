# Triage labels

五种 canonical role 与本地 Status 一一同名：needs-triage、needs-info、ready-for-agent、ready-for-human、wontfix。
每张已 triage 的票恰有一个 Category（bug/enhancement）和一个 Status。
Execution 独立记录 open/claimed/blocked/done，完成不增加第六个 triage 标签。
用户租卡/账号操作标 ready-for-human；缺少产品决定标 needs-info；纯技术依赖保留 ready-for-agent + blocked。
同一票出现互斥标签时先查最新明确用户决定，无依据则标冲突、记录待决策，不擅自消解。
